# data_pipeline.py
import pandas as pd
import numpy as np
import os
import time
import warnings
from pandas.errors import SettingWithCopyWarning # Import the necessary warning

# --- Configuration ---
RATS_URL = "https://data.cityofnewyork.us/resource/erm2-nwe9.csv?$select=unique_key,created_date,descriptor,borough,latitude,longitude&$where=complaint_type%3D%27Rodent%27&$limit=500000"
DUMPING_URL = "https://data.cityofnewyork.us/resource/erm2-nwe9.csv?$select=unique_key,created_date,descriptor,borough,latitude,longitude&$where=complaint_type%3D%27Illegal%20Dumping%27&$limit=500000"

RELIABLE_START_DATE = '2022-01-01'
CLEAN_DATE_COL = 'Created Date'
RAW_DATE_COL = 'created_date'
PRECISION = 4
SOCRATA_DATE_FORMAT = '%Y-%m-%dT%H:%M:%S.%f'

# --- Path Setup to handle execution from inside 'src' ---
# Get the absolute path to the directory containing this script (e.g., /path/to/project/src)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# The project root is one directory up (e.g., /path/to/project)
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# --- Output Configuration (CCDS compliant) ---
# Directory for model parameters and reports (at project root)
ARTIFACTS_DIR_NAME = 'artifacts' 
# Directory for processed data ready for modeling (at project root/data/processed)
DATA_PROCESSED_DIR_NAME = os.path.join('data')

# Full absolute paths for target directories
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, ARTIFACTS_DIR_NAME)
DATA_PROCESSED_DIR = os.path.join(PROJECT_ROOT, DATA_PROCESSED_DIR_NAME)

# File names and paths
FEATURES_FILE_NAME = 'df_features_final.csv'
THRESHOLD_FILE_NAME = 'rat_threshold.txt'
FEATURES_FILE_PATH = os.path.join(DATA_PROCESSED_DIR, FEATURES_FILE_NAME)
THRESHOLD_FILE_PATH = os.path.join(ARTIFACTS_DIR, THRESHOLD_FILE_NAME)

# Create necessary directories
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)

print(f"Model artifacts (threshold) will be saved in: {ARTIFACTS_DIR}/")
print(f"Processed feature data will be saved in: {DATA_PROCESSED_DIR}/")

# --- WARNING SUPPRESSION ---
# Suppress SettingWithCopyWarning for internal Pandas operations
warnings.filterwarnings("ignore", category=SettingWithCopyWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def load_raw_data():
    """Loads raw data from NYC Open Data APIs."""
    print("--- 0. Data Loading ---")
    try:
        # Use a small timeout or chunking for stability on large files, though direct read works here
        df_rats = pd.read_csv(RATS_URL)
        df_dumping = pd.read_csv(DUMPING_URL)
        print(f"Loaded {len(df_rats)} rat sightings and {len(df_dumping)} dumping incidents.")
        return df_rats, df_dumping
    except Exception as e:
        print(f"Failed to load data from API: {e}")
        return None, None


def clean_and_prepare(df_rats, df_dumping):
    """Performs cleaning, date conversion, and initial concatenation."""
    print("\n--- 1. Data Cleaning and Preparation ---")
    
    COL_MAP = {
        'latitude': 'Latitude', 'longitude': 'Longitude',
        'unique_key': 'Unique Key', 'borough': 'Borough'
    }

    df_rats.rename(columns=COL_MAP, inplace=True)
    df_dumping.rename(columns=COL_MAP, inplace=True)

    # Date Conversion
    for df in [df_rats, df_dumping]:
        if CLEAN_DATE_COL in df.columns:
            df.drop(columns=[CLEAN_DATE_COL], inplace=True)
        df.reset_index(drop=True, inplace=True)
        df[CLEAN_DATE_COL] = pd.to_datetime(
            df[RAW_DATE_COL].apply(str),
            format=SOCRATA_DATE_FORMAT,
            errors='coerce'
        )
        df.drop(columns=[RAW_DATE_COL], inplace=True)

    df_rats['incident_type'] = 'rat_sighting'
    df_dumping['incident_type'] = 'illegal_dumping'

    df_incidents_full = pd.concat([df_rats, df_dumping], ignore_index=True)
    df_incidents_full.dropna(subset=['Latitude', 'Longitude', CLEAN_DATE_COL], inplace=True)
    
    print(f"Total incidents processed (pre-filter): {len(df_incidents_full)} records.")
    return df_incidents_full


def aggregate_and_engineer(df_incidents_full):
    """Performs geospatial aggregation, time filtering, and feature engineering."""
    print("\n--- 2. Geospatial Aggregation and Filtering ---")
    
    # Geospatial Aggregation: Binning location coordinates
    df_incidents_full['Block_Lat'] = df_incidents_full['Latitude'].round(PRECISION)
    df_incidents_full['Block_Lon'] = df_incidents_full['Longitude'].round(PRECISION)
    df_incidents_full['Block_ID'] = df_incidents_full['Block_Lat'].astype(str) + "_" + df_incidents_full['Block_Lon'].astype(str)
    df_incidents_full['Month_Year'] = df_incidents_full[CLEAN_DATE_COL].dt.to_period('M')

    # Aggregate counts by Block and Month for both incident types
    df_aggregated_full = df_incidents_full.groupby(['Block_ID', 'Month_Year', 'incident_type']).size().unstack(fill_value=0)
    df_aggregated_full.columns = [f'count_{c}' for c in df_aggregated_full.columns]
    df_aggregated_full = df_aggregated_full.reset_index()

    # Ensure all count columns exist (in case one incident type is missing in the data chunk)
    for col in ['count_rat_sighting', 'count_illegal_dumping']:
        if col not in df_aggregated_full.columns:
            df_aggregated_full[col] = 0

    # Filtering for Reliable Data Window (for ML training)
    df_aggregated_ml = df_aggregated_full[
        pd.to_datetime(df_aggregated_full['Month_Year'].astype(str)) >= RELIABLE_START_DATE
    ].copy()
    
    df_aggregated_ml['Month_Start'] = df_aggregated_ml['Month_Year'].dt.to_timestamp()
    df_aggregated_ml = df_aggregated_ml.sort_values(by=['Block_ID', 'Month_Start']).reset_index(drop=True)
    print(f"Total aggregated records (Block-Months, FILTERED for ML): {len(df_aggregated_ml)}")
    
    print("\n--- 3. Creating Lagged Time-Series Features ---")

    # Lagged Features: Look at past incident counts
    df_aggregated_ml['lag_1_rat'] = df_aggregated_ml.groupby('Block_ID')['count_rat_sighting'].shift(1).fillna(0)
    df_aggregated_ml['lag_1_dumping'] = df_aggregated_ml.groupby('Block_ID')['count_illegal_dumping'].shift(1).fillna(0)
    df_aggregated_ml['lag_2_rat'] = df_aggregated_ml.groupby('Block_ID')['count_rat_sighting'].shift(2).fillna(0)
    df_aggregated_ml['lag_6_rat'] = df_aggregated_ml.groupby('Block_ID')['count_rat_sighting'].shift(6).fillna(0)
    
    # Rolling Average Feature (Chronic Issue Signal)
    df_aggregated_ml['lag_3_avg_rat'] = df_aggregated_ml.groupby('Block_ID')['count_rat_sighting'].shift(1).rolling(window=3, min_periods=1).mean().fillna(0)
    
    # Time Features
    df_aggregated_ml['month'] = df_aggregated_ml['Month_Start'].dt.month
    df_aggregated_ml['year'] = df_aggregated_ml['Month_Start'].dt.year
    
    df_features = df_aggregated_ml.copy()
    
    # --- START OF CHANGE 1: Ensure chronological and stable sorting before saving ---
    df_features = df_features.sort_values(by=['year', 'month', 'Block_ID']).reset_index(drop=True)
    # --- END OF CHANGE 1 ---

    # Define Target Variable (Predicting Next Month's Rat Risk)
    # Calculate the 90th percentile of rat sightings as the 'high risk' threshold
    RAT_THRESHOLD = df_features['count_rat_sighting'].quantile(0.90)
    
    # Shift next month's actual rat count back by one row within each block group
    df_features['next_month_rat_count'] = df_features.groupby('Block_ID')['count_rat_sighting'].shift(-1)
    
    # Drop the last record for each block since we don't know its target
    df_features.dropna(subset=['next_month_rat_count'], inplace=True)
    
    # Create the binary target variable (1 if next month is high risk, 0 otherwise)
    df_features['TARGET_HIGH_RISK'] = (df_features['next_month_rat_count'] >= RAT_THRESHOLD).astype(int)
    
    print(f"Target Threshold (90th percentile): {RAT_THRESHOLD:.0f} incidents/month")
    print(f"Target variable created. High-Risk records: {df_features['TARGET_HIGH_RISK'].sum()}")
    
    return df_features, RAT_THRESHOLD


if __name__ == '__main__':
    start_total = time.time()
    
    # 1. Load Data
    df_rats, df_dumping = load_raw_data()
    if df_rats is None:
        exit()

    # 2. Clean and Concatenate
    df_incidents_full = clean_and_prepare(df_rats, df_dumping)
    
    # 3. Aggregate, Engineer, and Target
    df_features_final, RAT_THRESHOLD = aggregate_and_engineer(df_incidents_full)

    # 4. Save Final Features DataFrame (to data/processed)
    df_features_final.to_csv(FEATURES_FILE_PATH, index=False)
    
    # 5. Save the RAT_THRESHOLD (to artifacts)
    with open(THRESHOLD_FILE_PATH, 'w') as f:
        f.write(str(RAT_THRESHOLD))
        
    end_total = time.time()
    print(f"\n--- Data Pipeline Complete ---")
    print(f"Final feature set saved to {FEATURES_FILE_PATH}")
    print(f"Threshold saved to {THRESHOLD_FILE_PATH}")
    print(f"Total pipeline time: {end_total - start_total:.2f} seconds.")