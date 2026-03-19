# model_predict.py
import pandas as pd
import numpy as np
import os
import time
import pickle
import warnings
import folium
from folium.plugins import HeatMap
from dateutil.relativedelta import relativedelta
from config import COUNT_FEATURES, TIME_FEATURES, BOROUGH_COL, TOP_N_MARKERS
from utils import get_logger, parse_block_id

logger = get_logger(__name__)

# --- Path Setup ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

ARTIFACTS_DIR_ROOT = os.path.join(PROJECT_ROOT, 'artifacts')
DATA_PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
MODEL_DIR = os.path.join(ARTIFACTS_DIR_ROOT, 'models')
REPORT_DIR = os.path.join(ARTIFACTS_DIR_ROOT, 'reports')
os.makedirs(REPORT_DIR, exist_ok=True)

FEATURES_FILE = 'df_features_final.csv'
ALL_BASE_FEATURES = COUNT_FEATURES + TIME_FEATURES + [BOROUGH_COL]

# Narrow, message-specific warning suppressions only (M-3 / M-4 pattern)
warnings.filterwarnings("ignore", category=UserWarning, message="X does not have valid feature names")
warnings.filterwarnings("ignore", category=UserWarning, message="The `max_val` parameter is no longer necessary.")


def load_deployment_artifacts():
    """Loads necessary data and artifacts for live prediction."""
    logger.info("--- 1. Loading Deployment Artifacts (Model, Scaler, Data, Training Columns) ---")

    artifacts = {}
    try:
        with open(os.path.join(MODEL_DIR, 'best_model.pkl'), 'rb') as f:
            artifacts['model'] = pickle.load(f)

        with open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'rb') as f:
            artifacts['scaler'] = pickle.load(f)

        with open(os.path.join(MODEL_DIR, 'X_train_cols.pkl'), 'rb') as f:
            artifacts['training_columns'] = pickle.load(f)

        with open(os.path.join(MODEL_DIR, 'rat_threshold.pkl'), 'rb') as f:
            artifacts['rat_threshold'] = pickle.load(f)

        with open(os.path.join(MODEL_DIR, 'decision_threshold.pkl'), 'rb') as f:
            artifacts['decision_threshold'] = pickle.load(f)

        data_path = os.path.join(DATA_PROCESSED_DIR, FEATURES_FILE)
        df_features = pd.read_csv(data_path)
        df_features.dropna(subset=['Block_ID', BOROUGH_COL], inplace=True)
        artifacts['df_features'] = df_features

    except FileNotFoundError as e:
        logger.error(f"Required artifact not found: {e}.")
        logger.error(
            f"Check MODEL_DIR: {MODEL_DIR} and DATA_PROCESSED_DIR: {DATA_PROCESSED_DIR}."
        )
        logger.error("Ensure prior scripts were run successfully.")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred during artifact loading: {e}")
        return None

    logger.info("Artifacts loaded successfully.")
    return artifacts


def generate_proactive_predictions(artifacts):
    """
    Identifies the latest month, scales data, generates predictions,
    and filters the predicted high-risk blocks (hotspots).
    """
    logger.info("\n--- 2. Generating Proactive Inspection List ---")
    start_time = time.time()

    df_full_features = artifacts['df_features'].copy()
    best_model = artifacts['model']
    scaler = artifacts['scaler']
    training_columns = artifacts['training_columns']
    decision_threshold = artifacts['decision_threshold']

    # Isolate features for the latest month that has actual sighting/dumping data.
    # The most recent period(s) may be empty if the NYC API hasn't published data yet.
    monthly_activity = (
        df_full_features.groupby('Month_Year')['count_rat_sighting'].sum()
    )
    active_months = monthly_activity[monthly_activity > 0]
    if active_months.empty:
        logger.error("No months with rat sighting data found. Cannot generate predictions.")
        return pd.DataFrame(), "Unknown"
    LATEST_MONTH_YEAR_STR = active_months.index.max()

    # H-2: Simplified Month_Year parsing — data_pipeline.py writes Period strings like "2024-11"
    try:
        LATEST_MONTH_DATETIME = pd.Period(LATEST_MONTH_YEAR_STR, freq='M').to_timestamp().to_pydatetime()
    except Exception:
        logger.warning(
            f"Could not parse Month_Year '{LATEST_MONTH_YEAR_STR}' as Period. Falling back."
        )
        LATEST_MONTH_DATETIME = pd.to_datetime(LATEST_MONTH_YEAR_STR).to_pydatetime()

    df_live_features = df_full_features[
        df_full_features['Month_Year'] == LATEST_MONTH_YEAR_STR
    ].copy()

    # H-3: Validate lag features before computing interaction term
    if df_live_features[['lag_1_rat', 'lag_1_dumping']].isnull().any().any():
        logger.warning(
            "NaN values found in lag features before interaction computation. Filling with 0."
        )
    df_live_features.loc[:, 'dumping_rat_interaction'] = (
        df_live_features['lag_1_rat'].fillna(0) * df_live_features['lag_1_dumping'].fillna(0)
    )

    # Prepare X_live
    X_live = df_live_features[ALL_BASE_FEATURES].copy()

    # M-4: Use drop_first=True to match model_train.py's OHE encoding
    X_live = pd.get_dummies(X_live, columns=[BOROUGH_COL], drop_first=True, prefix='Borough')

    if BOROUGH_COL in X_live.columns:
        X_live.drop(BOROUGH_COL, axis=1, inplace=True)

    # Scale count features
    X_live_scaled = X_live.copy()
    X_live_scaled[COUNT_FEATURES] = scaler.transform(X_live[COUNT_FEATURES])

    # Align columns to training feature order
    X_live_final_df = X_live_scaled.reindex(columns=training_columns, fill_value=0)
    X_live_final = X_live_final_df.values

    # Apply the recall-tuned decision threshold from training
    live_pred_proba = best_model.predict_proba(X_live_final)[:, 1]
    live_pred = (live_pred_proba >= decision_threshold).astype(int)

    df_live_predictions = df_live_features.copy()
    df_live_predictions.loc[:, 'Predicted_Risk'] = live_pred
    df_live_predictions.loc[:, 'High_Risk_Prob'] = live_pred_proba

    # H-1: Use parse_block_id utility instead of duplicated lambda
    lats, lons = zip(*df_live_predictions['Block_ID'].map(parse_block_id))
    df_live_predictions.loc[:, 'Lat'] = lats
    df_live_predictions.loc[:, 'Lon'] = lons

    # Determine prediction month string (next month's risk)
    prediction_month_dt = LATEST_MONTH_DATETIME + relativedelta(months=1)
    prediction_month_str = prediction_month_dt.strftime('%B %Y').upper()

    df_hotspots = df_live_predictions[df_live_predictions['Predicted_Risk'] == 1].copy()

    end_time = time.time()
    logger.info(f"Prediction Complete. Time taken: {end_time - start_time:.2f} seconds.")
    logger.info(
        f"Prediction for {prediction_month_str}: **{len(df_hotspots)}** high-risk blocks identified."
    )

    return df_hotspots, prediction_month_str


def generate_deployment_map(df_hotspots, prediction_month_str):
    """Generates the interactive HeatMap deployment map with top markers."""
    logger.info("\n--- 3. Generating Live Deployment Map ---")

    if not df_hotspots.empty:
        map_center_live = [df_hotspots['Lat'].mean(), df_hotspots['Lon'].mean()]
    else:
        map_center_live = [40.730610, -73.935242]

    m_live = folium.Map(
        location=map_center_live,
        zoom_start=11,
        tiles="cartodbpositron",
        width='100%',
        height=650
    )

    if not df_hotspots.empty:
        HeatMap(
            data=df_hotspots[['Lat', 'Lon', 'High_Risk_Prob']].values,
            radius=15,
            min_opacity=0.4
        ).add_to(m_live)

    df_top_hotspots = df_hotspots.sort_values(by='High_Risk_Prob', ascending=False).head(TOP_N_MARKERS)

    def create_popup(row):
        return f"""
        <h4>{prediction_month_str} RISK (Prob: {row['High_Risk_Prob']:.2f})</h4>
        ---
        <b>Block ID:</b> {row['Block_ID']}
        <b>Acute Risk (Lag 1 Rat):</b> {row['lag_1_rat']:.0f}
        <b>Chronic Issue (Lag 3 Avg Rat):</b> {row['lag_3_avg_rat']:.2f}
        <b>Interaction Term:</b> {row['dumping_rat_interaction']:.2f}
        """

    for _, row in df_top_hotspots.iterrows():
        folium.CircleMarker(
            location=[row['Lat'], row['Lon']],
            radius=5,
            color='darkorange',
            fill=True,
            fill_color='orange',
            fill_opacity=0.9,
            popup=folium.Popup(create_popup(row), max_width=300)
        ).add_to(m_live)

    legend_html = f'''
        <div style="position: fixed;
                    bottom: 50px; right: 50px; width: 280px; height: 180px;
                    border:2px solid grey; z-index:9999; font-size:14px;
                    background-color: white; opacity: 0.9; padding: 5px;">
            &nbsp; <b>{prediction_month_str} Inspection Targets</b> <br>
            <i style="background:orange; color:orange; border-radius:50%; width:10px; height:10px; display:inline-block; margin-left: 10px;"></i>
            <span style="margin-left: 5px;">Top {len(df_top_hotspots)} Priority Blocks (Inspection List)</span> <br>
            <br>
            &nbsp; HeatMap Risk Density<br>
            <div style="margin: 5px 10px; width: 80%; height: 20px;
                        background: linear-gradient(to right, blue, cyan, lime, yellow, red);">
            </div>
            <table style="width: 80%; margin: 0 10px; text-align: center;">
              <tr>
                <td style="width: 50%;">Low Risk</td>
                <td style="width: 50%;">High Risk</td>
              </tr>
            </table>
        </div>
        '''

    m_live.get_root().html.add_child(folium.Element(legend_html))

    map_filename = os.path.join(REPORT_DIR, 'risk_map_deployment.html')
    m_live.save(map_filename)
    logger.info(f"Saved live inspection map to {map_filename}")


def export_inspection_list(df_hotspots, prediction_month_str):
    """Exports the top risk list as a CSV for reporting."""
    if df_hotspots.empty:
        logger.info("No high-risk blocks identified to export.")
        return

    df_export = df_hotspots.sort_values(
        by='High_Risk_Prob', ascending=False
    ).head(TOP_N_MARKERS).copy()

    df_export = df_export[[
        'Block_ID', 'Lat', 'Lon', 'High_Risk_Prob',
        'lag_1_rat', 'lag_1_dumping', 'lag_3_avg_rat', 'dumping_rat_interaction'
    ]].copy()

    df_export.rename(columns={
        'High_Risk_Prob': 'Risk Score (P=High)',
        'lag_1_rat': 'Prior Rat Count (1 Month)',
        'lag_1_dumping': 'Prior Dumping Count (1 Month)',
        'lag_3_avg_rat': 'Prior Rat Avg (3 Month)',
        'dumping_rat_interaction': 'Dumping-Rat Interaction'
    }, inplace=True)

    df_export.insert(0, 'Rank', range(1, len(df_export) + 1))

    csv_filename = os.path.join(
        REPORT_DIR, f'inspection_list_{prediction_month_str.replace(" ", "_")}.csv'
    )
    df_export.to_csv(csv_filename, index=False)
    logger.info(f"Saved inspection list CSV to {csv_filename}")


if __name__ == '__main__':
    start_total = time.time()

    artifacts = load_deployment_artifacts()
    if artifacts is None:
        exit()

    df_hotspots, prediction_month_str = generate_proactive_predictions(artifacts)

    if not df_hotspots.empty:
        generate_deployment_map(df_hotspots, prediction_month_str)
        export_inspection_list(df_hotspots, prediction_month_str)
    else:
        logger.info("\nSkipping map and CSV export as no high-risk blocks were identified.")

    end_total = time.time()
    logger.info(f"\n--- Deployment Prediction Complete ---")
    logger.info(f"Total prediction time: {end_total - start_total:.2f} seconds.")
