# data_pipeline.py
import pandas as pd
import numpy as np
import os
import time
from config import (
    API_LIMIT, RELIABLE_START_DATE, COORDINATE_PRECISION, SOCRATA_DATE_FORMAT
)
from utils import get_logger

logger = get_logger(__name__)

# --- Configuration ---
RATS_URL = (
    "https://data.cityofnewyork.us/resource/erm2-nwe9.csv"
    "?$select=unique_key,created_date,descriptor,borough,latitude,longitude"
    f"&$where=complaint_type%3D%27Rodent%27&$limit={API_LIMIT}"
)
DUMPING_URL = (
    "https://data.cityofnewyork.us/resource/erm2-nwe9.csv"
    "?$select=unique_key,created_date,descriptor,borough,latitude,longitude"
    f"&$where=complaint_type%3D%27Illegal%20Dumping%27&$limit={API_LIMIT}"
)

CLEAN_DATE_COL = 'Created Date'
RAW_DATE_COL = 'created_date'

# --- Path Setup ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, 'artifacts')
DATA_PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
FEATURES_FILE_PATH = os.path.join(DATA_PROCESSED_DIR, 'df_features_final.csv')

os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)

logger.info(f"Processed feature data will be saved in: {DATA_PROCESSED_DIR}/")


def load_raw_data():
    """Loads raw data from NYC Open Data APIs."""
    logger.info("--- 0. Data Loading ---")
    try:
        df_rats = pd.read_csv(RATS_URL)
        df_dumping = pd.read_csv(DUMPING_URL)

        # H-4: Detect API truncation
        if len(df_rats) >= API_LIMIT:
            logger.warning(
                f"Rat sightings hit API limit ({API_LIMIT} rows). Data may be truncated."
            )
        if len(df_dumping) >= API_LIMIT:
            logger.warning(
                f"Illegal dumping hit API limit ({API_LIMIT} rows). Data may be truncated."
            )

        logger.info(f"Loaded {len(df_rats)} rat sightings and {len(df_dumping)} dumping incidents.")
        return df_rats, df_dumping
    except Exception as e:
        logger.error(f"Failed to load data from API: {e}")
        return None, None


def clean_and_prepare(df_rats, df_dumping):
    """Performs cleaning, date conversion, and initial concatenation."""
    logger.info("\n--- 1. Data Cleaning and Preparation ---")

    COL_MAP = {
        'latitude': 'Latitude', 'longitude': 'Longitude',
        'unique_key': 'Unique Key', 'borough': 'Borough'
    }

    df_rats.rename(columns=COL_MAP, inplace=True)
    df_dumping.rename(columns=COL_MAP, inplace=True)

    for df in [df_rats, df_dumping]:
        if CLEAN_DATE_COL in df.columns:
            df.drop(columns=[CLEAN_DATE_COL], inplace=True)
        df.reset_index(drop=True, inplace=True)
        df.loc[:, CLEAN_DATE_COL] = pd.to_datetime(
            df[RAW_DATE_COL].apply(str),
            format=SOCRATA_DATE_FORMAT,
            errors='coerce'
        )
        df.drop(columns=[RAW_DATE_COL], inplace=True)

    df_rats['incident_type'] = 'rat_sighting'
    df_dumping['incident_type'] = 'illegal_dumping'

    df_incidents_full = pd.concat([df_rats, df_dumping], ignore_index=True)
    df_incidents_full.dropna(subset=['Latitude', 'Longitude', CLEAN_DATE_COL], inplace=True)

    logger.info(f"Total incidents processed (pre-filter): {len(df_incidents_full)} records.")
    return df_incidents_full


def aggregate_and_engineer(df_incidents_full):
    """Performs geospatial aggregation, time filtering, and feature engineering.

    C-1: Does NOT compute the rat threshold or TARGET_HIGH_RISK column.
    The threshold is computed from training data only in model_train.py.
    Returns df_features with raw next_month_rat_count intact.
    """
    logger.info("\n--- 2. Geospatial Aggregation and Filtering ---")

    df_incidents_full['Block_Lat'] = df_incidents_full['Latitude'].round(COORDINATE_PRECISION)
    df_incidents_full['Block_Lon'] = df_incidents_full['Longitude'].round(COORDINATE_PRECISION)
    df_incidents_full['Block_ID'] = (
        df_incidents_full['Block_Lat'].astype(str) + "_" +
        df_incidents_full['Block_Lon'].astype(str)
    )
    df_incidents_full['Month_Year'] = df_incidents_full[CLEAN_DATE_COL].dt.to_period('M')

    df_aggregated_full = df_incidents_full.groupby(
        ['Block_ID', 'Borough', 'Month_Year', 'incident_type']
    ).size().unstack(fill_value=0)
    df_aggregated_full.columns = [f'count_{c}' for c in df_aggregated_full.columns]
    df_aggregated_full = df_aggregated_full.reset_index()

    for col in ['count_rat_sighting', 'count_illegal_dumping']:
        if col not in df_aggregated_full.columns:
            df_aggregated_full[col] = 0

    df_aggregated_ml = df_aggregated_full[
        pd.to_datetime(df_aggregated_full['Month_Year'].astype(str)) >= RELIABLE_START_DATE
    ].copy()

    df_aggregated_ml['Month_Start'] = df_aggregated_ml['Month_Year'].dt.to_timestamp()
    df_aggregated_ml = df_aggregated_ml.sort_values(
        by=['Block_ID', 'Month_Start']
    ).reset_index(drop=True)
    logger.info(f"Total aggregated records (Block-Months, FILTERED for ML): {len(df_aggregated_ml)}")

    logger.info("\n--- 3. Creating Lagged Time-Series Features ---")

    df_aggregated_ml['lag_1_rat'] = (
        df_aggregated_ml.groupby('Block_ID')['count_rat_sighting'].shift(1).fillna(0)
    )
    df_aggregated_ml['lag_1_dumping'] = (
        df_aggregated_ml.groupby('Block_ID')['count_illegal_dumping'].shift(1).fillna(0)
    )
    df_aggregated_ml['lag_2_rat'] = (
        df_aggregated_ml.groupby('Block_ID')['count_rat_sighting'].shift(2).fillna(0)
    )
    df_aggregated_ml['lag_6_rat'] = (
        df_aggregated_ml.groupby('Block_ID')['count_rat_sighting'].shift(6).fillna(0)
    )
    df_aggregated_ml['lag_3_avg_rat'] = (
        df_aggregated_ml.groupby('Block_ID')['count_rat_sighting']
        .shift(1).rolling(window=3, min_periods=1).mean().fillna(0)
    )
    df_aggregated_ml['lag_12_avg_rat'] = (
        df_aggregated_ml.groupby('Block_ID')['count_rat_sighting']
        .shift(1).rolling(window=12, min_periods=1).mean().fillna(0)
    )
    df_aggregated_ml['dumping_rat_interaction'] = (
        df_aggregated_ml['lag_1_rat'] * df_aggregated_ml['lag_1_dumping']
    )

    df_aggregated_ml['month'] = df_aggregated_ml['Month_Start'].dt.month
    df_aggregated_ml['year'] = df_aggregated_ml['Month_Start'].dt.year

    df_features = df_aggregated_ml.copy()
    df_features = df_features.sort_values(
        by=['Block_ID', 'Month_Start']
    ).reset_index(drop=True)

    # C-1: Keep next_month_rat_count raw; threshold will be computed in model_train.py
    df_features['next_month_rat_count'] = (
        df_features.groupby('Block_ID')['count_rat_sighting'].shift(-1)
    )
    df_features.dropna(subset=['next_month_rat_count'], inplace=True)

    logger.info(f"Feature engineering complete. {len(df_features)} block-month records.")
    return df_features


if __name__ == '__main__':
    start_total = time.time()

    df_rats, df_dumping = load_raw_data()
    if df_rats is None:
        exit()

    df_incidents_full = clean_and_prepare(df_rats, df_dumping)
    df_features_final = aggregate_and_engineer(df_incidents_full)

    df_features_final.to_csv(FEATURES_FILE_PATH, index=False)

    end_total = time.time()
    logger.info(f"\n--- Data Pipeline Complete ---")
    logger.info(f"Final feature set saved to {FEATURES_FILE_PATH}")
    logger.info(f"Total pipeline time: {end_total - start_total:.2f} seconds.")
