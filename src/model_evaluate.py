# model_evaluate.py
import pandas as pd
import numpy as np
import os
import time
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import MarkerCluster
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from config import COUNT_FEATURES, TIME_FEATURES, BOROUGH_COL, TARGET, RELIABLE_START_DATE
from utils import get_logger, parse_block_id

logger = get_logger(__name__)

# --- Path Setup ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

ARTIFACTS_DIR_ROOT = os.path.join(PROJECT_ROOT, 'artifacts')
DATA_PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
FEATURES_FILE_PATH = os.path.join(DATA_PROCESSED_DIR, 'df_features_final.csv')

MODEL_DIR = os.path.join(ARTIFACTS_DIR_ROOT, 'models')
REPORT_DIR = os.path.join(ARTIFACTS_DIR_ROOT, 'reports')
SPLIT_DIR = os.path.join(ARTIFACTS_DIR_ROOT, 'data_splits')


def load_cluster_icon_js() -> str:
    """Loads the Folium cluster icon JavaScript from the external file (M-5)."""
    js_path = os.path.join(os.path.dirname(__file__), 'folium_icons.js')
    with open(js_path) as f:
        return f.read()


def load_artifacts():
    """Loads necessary data and artifacts for evaluation."""
    logger.info("--- 1. Loading Artifacts for Evaluation ---")

    artifacts = {}

    try:
        df_features = pd.read_csv(FEATURES_FILE_PATH)
        df_test_performance = pd.read_csv(os.path.join(SPLIT_DIR, 'X_test_features.csv'))

        with open(os.path.join(MODEL_DIR, 'best_model.pkl'), 'rb') as f:
            artifacts['best_model'] = pickle.load(f)

        # C-1: Load threshold computed from training data only
        with open(os.path.join(MODEL_DIR, 'rat_threshold.pkl'), 'rb') as f:
            artifacts['rat_threshold'] = pickle.load(f)

    except FileNotFoundError as e:
        logger.error(
            f"Required artifact not found: {e}. "
            "Ensure data_pipeline.py and model_train.py were run successfully."
        )
        return None

    artifacts['df_features'] = df_features
    artifacts['df_test_performance'] = df_test_performance

    artifacts['df_features'].dropna(subset=['Block_ID'], inplace=True)
    artifacts['df_test_performance'].dropna(subset=['Block_ID'], inplace=True)

    # H-1: Use parse_block_id utility instead of duplicated lambda
    lats, lons = zip(*artifacts['df_features']['Block_ID'].map(parse_block_id))
    artifacts['df_features'].loc[:, 'Lat'] = lats
    artifacts['df_features'].loc[:, 'Lon'] = lons

    lats, lons = zip(*artifacts['df_test_performance']['Block_ID'].map(parse_block_id))
    artifacts['df_test_performance'].loc[:, 'Lat'] = lats
    artifacts['df_test_performance'].loc[:, 'Lon'] = lons

    # C-1: Recompute TARGET_HIGH_RISK for EDA using the train-derived threshold
    rat_threshold = artifacts['rat_threshold']
    artifacts['df_features'][TARGET] = (
        artifacts['df_features']['next_month_rat_count'] >= rat_threshold
    ).astype(int)

    # C-3: Load split_index from file written by model_train.py
    with open(os.path.join(SPLIT_DIR, 'split_index.txt'), 'r') as f:
        artifacts['split_index'] = int(f.read().strip())

    logger.info(f"Artifacts loaded. Threshold: {rat_threshold:.1f}, Split index: {artifacts['split_index']}")
    return artifacts


def generate_eda_charts(df_features):
    """Generates and saves standard Exploratory Data Analysis charts."""
    logger.info("\n--- 2. Generating and Saving EDA Charts ---")

    df_aggregated_full = df_features.groupby(
        ['Block_ID', 'Month_Year', 'count_rat_sighting', 'count_illegal_dumping']
    ).first().reset_index()

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    axes = axes.flatten()

    # Subplot 1: Time series
    time_series_data_full = df_aggregated_full.groupby('Month_Year')[
        ['count_rat_sighting', 'count_illegal_dumping']
    ].sum().reset_index()
    time_series_data_full['date'] = pd.to_datetime(time_series_data_full['Month_Year'].astype(str))

    ax_ts = axes[0]
    color_rat = 'tab:red'
    ax_ts.plot(time_series_data_full['date'], time_series_data_full['count_rat_sighting'],
               color=color_rat, label='Rat Sightings (Count)', marker='o', linewidth=1)
    ax_ts.set_ylabel('Rat Sightings (Count)', color=color_rat, fontsize=12)
    ax_ts.tick_params(axis='y', labelcolor=color_rat)
    ax_ts_2 = ax_ts.twinx()
    color_dumping = 'tab:blue'
    ax_ts_2.plot(time_series_data_full['date'], time_series_data_full['count_illegal_dumping'],
                 color=color_dumping, label='Illegal Dumping (Count)', linestyle='--', marker='x', linewidth=1)
    ax_ts_2.set_ylabel('Illegal Dumping (Count)', color=color_dumping, fontsize=12)
    ax_ts_2.tick_params(axis='y', labelcolor=color_dumping)
    ax_ts.set_title(f'1. Temporal Trend of Incidents (Filtered from {RELIABLE_START_DATE})', fontsize=14, color='darkred')
    ax_ts.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m'))
    ax_ts.tick_params(axis='x', rotation=45)

    # Subplot 2: Correlation matrix
    full_corr_cols = COUNT_FEATURES + TIME_FEATURES + [TARGET, BOROUGH_COL]
    df_corr_matrix = df_features[df_features[TARGET].notna()].copy()
    df_corr_matrix_raw = df_corr_matrix[df_corr_matrix.columns.intersection(full_corr_cols)]
    corr_matrix = df_corr_matrix_raw.select_dtypes(include=np.number).corr()

    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f",
                linewidths=.5, ax=axes[1], cbar=False)
    axes[1].set_title('2. Feature Correlation Matrix (Raw Features)', fontsize=14)

    # Subplot 3: Monthly seasonal trend
    monthly_trends = df_features.groupby('month')[
        ['count_rat_sighting', 'count_illegal_dumping']
    ].sum().reset_index()
    color = 'tab:red'
    axes[2].set_ylabel('Total Rat Sightings', color=color, fontsize=12)
    axes[2].plot(monthly_trends['month'], monthly_trends['count_rat_sighting'],
                 color=color, label='Rat Sightings')
    ax2_sub3 = axes[2].twinx()
    color = 'tab:blue'
    ax2_sub3.set_ylabel('Total Illegal Dumping', color=color, fontsize=12)
    ax2_sub3.plot(monthly_trends['month'], monthly_trends['count_illegal_dumping'],
                  color=color, linestyle='--', label='Illegal Dumping')
    axes[2].set_title('3. Total Monthly Incidents (Seasonal Trend)', fontsize=14)

    # Subplot 4: Prior dumping vs expected rat sightings
    dumping_mean_rat = df_features.groupby('lag_1_dumping')['next_month_rat_count'].mean().reset_index()
    dumping_mean_rat = dumping_mean_rat[dumping_mean_rat['lag_1_dumping'] <= 10]
    sns.barplot(x='lag_1_dumping', y='next_month_rat_count', data=dumping_mean_rat,
                ax=axes[3], palette='viridis', legend=False)
    axes[3].set_title('4. Expected Rat Sightings vs. Prior Dumping (Neglect Gate)', fontsize=14)

    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, 'eda_charts.png'))
    plt.close(fig)
    logger.info(f"Saved EDA charts to {REPORT_DIR}/eda_charts.png")


def generate_feature_importance(best_model):
    """Generates and saves the feature importance chart. C-7: model-agnostic."""
    logger.info("\n--- 3. Generating and Saving Feature Importance ---")

    # C-7: Model-agnostic feature name extraction
    if hasattr(best_model, 'feature_names_in_'):
        feature_names = best_model.feature_names_in_
    elif hasattr(best_model, 'get_booster'):
        feature_names = list(best_model.get_booster().feature_names)
    else:
        logger.warning("Could not reliably extract feature names from the model.")
        return

    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
    elif hasattr(best_model, 'get_booster'):
        importance_scores = best_model.get_booster().get_score(importance_type='gain')
        importances = np.array([importance_scores.get(f, 0.0) for f in feature_names])
    else:
        logger.warning("Model does not have a feature_importances_ attribute.")
        return

    feature_importances = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    feature_importances.to_csv(os.path.join(REPORT_DIR, 'feature_importance.csv'))

    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_importances.values, y=feature_importances.index, palette='viridis')
    plt.title(f'Feature Importance from Best Model ({type(best_model).__name__})')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, 'feature_importance.png'))
    plt.close()
    logger.info(f"Saved feature importance chart and data to {REPORT_DIR}/")


def calculate_metrics_by_borough(df_test_performance, df_features):
    """Calculates Precision, Recall, F1-Score, and Accuracy for each Borough."""
    logger.info("\n--- 4. Calculating Performance Metrics by Borough ---")

    borough_map = df_features[['Block_ID', BOROUGH_COL]].drop_duplicates(subset=['Block_ID'])

    df_test_performance = df_test_performance.rename(
        columns={TARGET: 'y_true', 'Predicted_Risk': 'y_pred'}
    )

    df_performance_with_borough = df_test_performance.merge(
        borough_map, on='Block_ID', how='left'
    )

    def calculate_group_metrics(group):
        y_true = group['y_true']
        y_pred = group['y_pred']
        return pd.Series({
            'Count': len(group),
            'True Positives': sum((y_true == 1) & (y_pred == 1)),
            'False Positives': sum((y_true == 0) & (y_pred == 1)),
            'False Negatives': sum((y_true == 1) & (y_pred == 0)),
            'Accuracy': accuracy_score(y_true, y_pred),
            'Precision (Risk)': precision_score(y_true, y_pred, zero_division=0),
            'Recall (Risk)': recall_score(y_true, y_pred, zero_division=0),
            'F1-Score (Risk)': f1_score(y_true, y_pred, zero_division=0)
        })

    metrics_by_borough = df_performance_with_borough.groupby(BOROUGH_COL).apply(
        calculate_group_metrics
    ).sort_values(by='F1-Score (Risk)', ascending=False).reset_index()

    metrics_by_borough.to_csv(os.path.join(REPORT_DIR, 'metrics_by_borough.csv'), index=False)

    logger.info(f"Saved Borough metrics to {REPORT_DIR}/metrics_by_borough.csv")
    logger.info("\nBorough Performance (F1-Score sorted):")
    logger.info(
        metrics_by_borough[
            [BOROUGH_COL, 'F1-Score (Risk)', 'Precision (Risk)', 'Recall (Risk)', 'Count']
        ].round(3).to_markdown(index=False)
    )

    return df_test_performance, metrics_by_borough


def generate_dual_map(artifacts, n_samples=5000):
    """Generates the dual map HTML: Split Map and Performance Map."""
    logger.info("\n--- 5. Generating Dual Train/Test/Performance Map ---")

    # C-2: Stable sort matching model_train.py's sort order
    df_features = artifacts['df_features'].sort_values(
        by=['year', 'month', 'Block_ID'], ascending=True
    )
    df_test_performance_raw = artifacts['df_test_performance']
    split_index = artifacts['split_index']  # C-3: from file, not recomputed

    df_train_locations = df_features.iloc[:split_index].copy()

    df_test_performance = df_test_performance_raw.copy().rename(
        columns={TARGET: 'y_true', 'Predicted_Risk': 'y_pred'}
    )
    df_test_performance.loc[:, 'Outcome'] = np.where(
        (df_test_performance['y_true'] == 1) & (df_test_performance['y_pred'] == 1), 'TP',
        np.where(
            (df_test_performance['y_true'] == 0) & (df_test_performance['y_pred'] == 1), 'FP',
            np.where(
                (df_test_performance['y_true'] == 1) & (df_test_performance['y_pred'] == 0), 'FN',
                'TN'
            )
        )
    )
    map_center = [df_features['Lat'].mean(), df_features['Lon'].mean()]

    train_sample_size = int(n_samples * 0.8)
    test_sample_size = n_samples - train_sample_size

    np.random.seed(42)

    df_train_sample = df_train_locations.sample(
        n=min(len(df_train_locations), train_sample_size), replace=False
    )[['Lat', 'Lon']]

    df_test_sample_performance = df_test_performance.sample(
        n=min(len(df_test_performance), test_sample_size), replace=False
    )

    df_test_sample_m1 = df_test_sample_performance[['Lat', 'Lon']]
    actual_train_sample_size = len(df_train_sample)
    actual_test_sample_size = len(df_test_sample_m1)

    # Map 1: Train/Test Split
    m1 = folium.Map(location=map_center, zoom_start=11, tiles='cartodbpositron')

    for _, row in df_train_sample.iterrows():
        folium.Circle(
            location=[row['Lat'], row['Lon']], radius=5, color='blue',
            fill=True, fill_color='blue', fill_opacity=0.6, tooltip="Train Point"
        ).add_to(m1)
    for _, row in df_test_sample_m1.iterrows():
        folium.Circle(
            location=[row['Lat'], row['Lon']], radius=5, color='red',
            fill=True, fill_color='red', fill_opacity=0.7, tooltip="Test Point"
        ).add_to(m1)

    legend_html_m1 = f'''
        <div style="position: fixed; bottom: 50px; left: 50px; width: 170px; height: 100px; border:2px solid grey; z-index:9999; font-size:12px; background-color: white; opacity: 0.9;">
          &nbsp; <b>Map 1: Data Split (80/20)</b> <br>
          &nbsp; <i style="color:blue" class="fa fa-circle fa-1x"></i> Training Data ({actual_train_sample_size} points) <br>
          &nbsp; <i style="color:red" class="fa fa-circle fa-1x"></i> Test Data ({actual_test_sample_size} points) <br>
        </div>
        '''
    m1.get_root().html.add_child(folium.Element(legend_html_m1))

    # Map 2: Performance
    tn_count = len(df_test_sample_performance[df_test_sample_performance['Outcome'] == 'TN'])
    tp_count = len(df_test_sample_performance[df_test_sample_performance['Outcome'] == 'TP'])
    fp_count = len(df_test_sample_performance[df_test_sample_performance['Outcome'] == 'FP'])
    fn_count = len(df_test_sample_performance[df_test_sample_performance['Outcome'] == 'FN'])
    total_performance_points_m2 = tn_count + tp_count + fp_count + fn_count

    # M-5: Load cluster icon JS from external file
    js_icon_create_function = load_cluster_icon_js()

    m2 = folium.Map(location=map_center, zoom_start=11, tiles='cartodbpositron')
    dominance_cluster = MarkerCluster(icon_create_function=js_icon_create_function).add_to(m2)

    color_map = {'TP': 'green', 'FP': 'red', 'FN': 'orange', 'TN': 'gray'}
    icon_map = {'TP': 'check', 'FP': 'times', 'FN': 'exclamation-triangle', 'TN': 'circle'}

    for _, row in df_test_sample_performance.iterrows():
        outcome = row['Outcome']
        color = color_map.get(outcome)

        if outcome == 'TN':
            folium.Circle(
                location=[row['Lat'], row['Lon']], radius=3, color='gray',
                fill=True, fill_color='gray', fill_opacity=0.5
            ).add_to(dominance_cluster)
        else:
            icon_type = icon_map.get(outcome)
            icon_html = f"""<div style="text-align: center;" data-outcome="{outcome}"><i class="fa fa-{icon_type} fa-2x" style="color:{color}"></i></div>"""
            icon = folium.DivIcon(html=icon_html, icon_size=(24, 24))
            folium.Marker(
                location=[row['Lat'], row['Lon']], icon=icon, tooltip=f"Outcome: {outcome}"
            ).add_to(dominance_cluster)

    legend_icon_key = f'''
        <div style="position: fixed; bottom: 50px; left: 50px; width: 220px; height: 120px; border:2px solid grey; z-index:9999; font-size:12px; background-color: white; opacity: 0.9;">
          &nbsp; <b>MARKER ICON KEY</b> <br>
          &nbsp; <i style="color:green" class="fa fa-check fa-1x"></i> True Positive (Successful Risk) <br>
          &nbsp; <i style="color:red" class="fa fa-times fa-1x"></i> False Positive (Wasted Inspection) <br>
          &nbsp; <i style="color:orange" class="fa fa-exclamation-triangle fa-1x"></i> False Negative (Missed Risk) <br>
          &nbsp; <i style="color:gray" class="fa fa-circle fa-1x"></i> True Negative (Correctly Ignored) <br>
        </div>
        '''
    m2.get_root().html.add_child(folium.Element(legend_icon_key))

    legend_cluster_key = f'''
        <div style="position: fixed; bottom: 50px; right: 20px; width: 140px; height: 120px; border:2px solid grey; z-index:9999; font-size:12px; background-color: white; opacity: 0.9;">
          &nbsp; <b>CLUSTER COLORS</b> <br>
          &nbsp; <i style="color:green" class="fa fa-circle fa-1x"></i> TP Dominance <br>
          &nbsp; <i style="color:red" class="fa fa-circle fa-1x"></i> FP Dominance <br>
          &nbsp; <i style="color:orange" class="fa fa-circle fa-1x"></i> FN/Mixed Zone <br>
          &nbsp; <i style="color:gray" class="fa fa-circle fa-1x"></i> TN Dominance <br>
        </div>
        '''
    m2.get_root().html.add_child(folium.Element(legend_cluster_key))

    m1_html = m1._repr_html_()
    m2_html = m2._repr_html_()

    map1_title = f"Map 1: Chronological Train/Test Split (80/20 Sample: N={n_samples})"
    map2_title = f"Map 2: Model Performance on Test Set (N={actual_test_sample_size} Test Time-Steps)"
    map2_subtitle = f"Outcomes: TP:{tp_count}, FP:{fp_count}, FN:{fn_count}, TN:{tn_count}"

    final_html = f"""
    <div style="display: flex; justify-content: space-around; width: 100%;">
        <div style="flex-grow: 1; padding-right: 5px; min-width: 48%;">
            <h3>{map1_title}</h3>
            {m1_html}
        </div>
        <div style="flex-grow: 1; padding-left: 5px; min-width: 48%;">
            <h3>{map2_title}</h3>
            <p style="margin-top: -15px; margin-bottom: 5px; font-size: 14px; color: #555;">{map2_subtitle}</p>
            {m2_html}
        </div>
    </div>
    """

    with open(os.path.join(REPORT_DIR, 'dual_train_test_map.html'), 'w') as f:
        f.write(final_html)
    logger.info(f"Saved Dual Train/Test/Performance Map to {REPORT_DIR}/dual_train_test_map.html")


if __name__ == '__main__':
    start_time = time.time()

    artifacts = load_artifacts()
    if artifacts is None:
        exit()

    generate_eda_charts(artifacts['df_features'])

    df_test_performance_final, metrics_by_borough = calculate_metrics_by_borough(
        artifacts['df_test_performance'],
        artifacts['df_features']
    )

    artifacts['df_test_performance'] = df_test_performance_final

    generate_feature_importance(artifacts['best_model'])

    generate_dual_map(artifacts)

    end_time = time.time()
    logger.info(f"\n--- Evaluation Complete ---")
    logger.info(f"Total evaluation time: {end_time - start_time:.2f} seconds.")
