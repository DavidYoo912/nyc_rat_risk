import pandas as pd
import numpy as np
import os
import time
import pickle
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import MarkerCluster
from IPython.display import display

# --- Path Setup to handle execution from inside 'src' ---
# Get the absolute path to the directory containing this script (e.g., /path/to/project/src)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# The project root is one directory up (e.g., /path/to/project)
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# --- Configuration (Must match other scripts) ---
# FIX: Use absolute path construction relative to PROJECT_ROOT
ARTIFACTS_DIR_ROOT = os.path.join(PROJECT_ROOT, 'artifacts')
# New variable to point to where the raw features data is stored (as per data_pipeline)
DATA_PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data') 

# Organized Artifact Subdirectories for loading/saving
MODEL_DIR = os.path.join(ARTIFACTS_DIR_ROOT, 'models')
REPORT_DIR = os.path.join(ARTIFACTS_DIR_ROOT, 'reports')
SPLIT_DIR = os.path.join(ARTIFACTS_DIR_ROOT, 'data_splits')

# Standard Configuration
RELIABLE_START_DATE = '2022-01-01'
COUNT_FEATURES = ['lag_1_rat', 'lag_1_dumping', 'lag_2_rat', 'lag_6_rat', 'lag_3_avg_rat']
TIME_FEATURES = ['month', 'year']
FEATURES = COUNT_FEATURES + TIME_FEATURES
TARGET = 'TARGET_HIGH_RISK'

# --- WARNING SUPPRESSION ---
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
try:
    from pandas.errors import SettingWithCopyWarning
    warnings.filterwarnings("ignore", category=SettingWithCopyWarning)
except ImportError:
    pass


def load_artifacts():
    """Loads necessary data and artifacts for evaluation."""
    print("--- 1. Loading Artifacts for Evaluation ---")
    
    artifacts = {}
    
    # Define the path for the raw features file (Input from data_pipeline)
    FEATURES_FILE_PATH = os.path.join(DATA_PROCESSED_DIR, 'df_features_final.csv')
    
    try:
        # Load Features Data from the DATA_PROCESSED_DIR
        df_features = pd.read_csv(FEATURES_FILE_PATH)
        
        # Load Test Performance Data (saved in data_splits subdirectory)
        df_test_performance = pd.read_csv(os.path.join(SPLIT_DIR, 'X_test_features.csv'))
        
        # Load Split Index (saved in data_splits subdirectory)
        with open(os.path.join(SPLIT_DIR, 'split_index.txt'), 'r') as f:
            artifacts['split_index'] = int(f.read())
            
        # Load Model (saved in models subdirectory)
        with open(os.path.join(MODEL_DIR, 'best_model.pkl'), 'rb') as f:
            artifacts['best_model'] = pickle.load(f)
            
    except FileNotFoundError as e:
        # Corrected error message to show the absolute path that failed
        print(f"Error: Required artifact not found: {e}. Ensure data_pipeline.py and model_train.py were run successfully and files are in the correct subdirectories.")
        return None

    artifacts['df_features'] = df_features
    artifacts['df_test_performance'] = df_test_performance
    
    # --- FIX: Drop rows with missing Block_ID before attempting to split the string ---
    artifacts['df_features'].dropna(subset=['Block_ID'], inplace=True)
    artifacts['df_test_performance'].dropna(subset=['Block_ID'], inplace=True)
    # --- END FIX ---
    
    # Calculate Lat/Lon for mapping purposes (These lines are now safe)
    # Use .loc to avoid SettingWithCopyWarning
    artifacts['df_features'].loc[:, 'Lat'] = artifacts['df_features']['Block_ID'].apply(lambda x: float(x.split('_')[0]))
    artifacts['df_features'].loc[:, 'Lon'] = artifacts['df_features']['Block_ID'].apply(lambda x: float(x.split('_')[1]))

    artifacts['df_test_performance'].loc[:, 'Lat'] = artifacts['df_test_performance']['Block_ID'].apply(lambda x: float(x.split('_')[0]))
    artifacts['df_test_performance'].loc[:, 'Lon'] = artifacts['df_test_performance']['Block_ID'].apply(lambda x: float(x.split('_')[1]))
    
    print("Artifacts loaded successfully.")
    return artifacts


def generate_eda_charts(df_features):
    """Generates and saves standard Exploratory Data Analysis charts."""
    print("\n--- 2. Generating and Saving EDA Charts ---")
    
    # Use the full data from the pipeline to ensure the full history is plotted
    df_aggregated_full = df_features.groupby(['Block_ID', 'Month_Year', 'count_rat_sighting', 'count_illegal_dumping']).first().reset_index()
    
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    axes = axes.flatten()

    # --- SUBPLOT 1: Time Series of Counts (FILTERED DATA) ---
    time_series_data_full = df_aggregated_full.groupby('Month_Year')[
        ['count_rat_sighting', 'count_illegal_dumping']
    ].sum().reset_index()
    time_series_data_full['date'] = pd.to_datetime(time_series_data_full['Month_Year'].astype(str))
    
    ax_ts = axes[0]
    color_rat = 'tab:red'
    ax_ts.plot(time_series_data_full['date'], time_series_data_full['count_rat_sighting'], color=color_rat, label='Rat Sightings (Count)', marker='o', linewidth=1)
    ax_ts.set_ylabel('Rat Sightings (Count)', color=color_rat, fontsize=12)
    ax_ts.tick_params(axis='y', labelcolor=color_rat)
    ax_ts_2 = ax_ts.twinx()
    color_dumping = 'tab:blue'
    ax_ts_2.plot(time_series_data_full['date'], time_series_data_full['count_illegal_dumping'], color=color_dumping, label='Illegal Dumping (Count)', linestyle='--', marker='x', linewidth=1)
    ax_ts_2.set_ylabel('Illegal Dumping (Count)', color=color_dumping, fontsize=12)
    ax_ts_2.tick_params(axis='y', labelcolor=color_dumping)
    ax_ts.set_title(f'1. Temporal Trend of Incidents (Filtered from {RELIABLE_START_DATE})', fontsize=14, color='darkred')
    ax_ts.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m'))
    ax_ts.tick_params(axis='x', rotation=45)

    # --- SUBPLOT 2: Correlation Matrix ---
    corr_cols = FEATURES + [TARGET]
    # Filter the features data to remove rows that were dropped (where next_month_rat_count was NaN)
    df_corr_matrix = df_features[df_features[TARGET].notna()].copy()
    corr_matrix = df_corr_matrix[corr_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=axes[1], cbar=False)
    axes[1].set_title('2. Feature Correlation Matrix', fontsize=14)

    # --- SUBPLOT 3: Temporal Trend of Incidents (Monthly) ---
    monthly_trends = df_features.groupby('month')[['count_rat_sighting', 'count_illegal_dumping']].sum().reset_index()
    color = 'tab:red'
    axes[2].set_ylabel('Total Rat Sightings', color=color, fontsize=12)
    axes[2].plot(monthly_trends['month'], monthly_trends['count_rat_sighting'], color=color, label='Rat Sightings')
    ax2_sub3 = axes[2].twinx()
    color = 'tab:blue'
    ax2_sub3.set_ylabel('Total Illegal Dumping', color=color, fontsize=12)
    ax2_sub3.plot(monthly_trends['month'], monthly_trends['count_illegal_dumping'], color=color, linestyle='--', label='Illegal Dumping')
    axes[2].set_title(f'3. Total Monthly Incidents (Seasonal Trend)', fontsize=14)

    # --- SUBPLOT 4: Prior Dumping vs. Expected Rat Sightings ---
    dumping_mean_rat = df_features.groupby('lag_1_dumping')['next_month_rat_count'].mean().reset_index()
    dumping_mean_rat = dumping_mean_rat[dumping_mean_rat['lag_1_dumping'] <= 10]
    sns.barplot(x='lag_1_dumping', y='next_month_rat_count', data=dumping_mean_rat, ax=axes[3], palette='viridis', legend=False)
    axes[3].set_title(f'4. Expected Rat Sightings vs. Prior Dumping (Neglect Gate)', fontsize=14)

    plt.tight_layout()
    # Save the figure to the reports directory
    plt.savefig(os.path.join(REPORT_DIR, 'eda_charts.png'))
    plt.close(fig)
    print(f"Saved EDA charts to {REPORT_DIR}/eda_charts.png")


def generate_feature_importance(best_model):
    """Generates and saves the feature importance chart."""
    print("\n--- 3. Generating and Saving Feature Importance ---")
    
    feature_importances = pd.Series(
        best_model.feature_importances_,
        index=FEATURES
    ).sort_values(ascending=False)
    
    # Save data and chart to the reports directory
    feature_importances.to_csv(os.path.join(REPORT_DIR, 'feature_importance.csv'))

    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_importances.values, y=feature_importances.index, palette='viridis')
    plt.title('Feature Importance from Best Random Forest Model')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, 'feature_importance.png'))
    plt.close()
    print(f"Saved feature importance chart and data to {REPORT_DIR}/")


def generate_dual_map(artifacts, n_samples=5000):
    """
    Generates the dual map HTML string: Split Map and Performance Map, including legends.
    """
    print("\n--- 4. Generating Dual Train/Test/Performance Map ---")
    
    df_features = artifacts['df_features']
    df_test_performance = artifacts['df_test_performance']
    split_index = artifacts['split_index']
    
    # --- Data Setup for Maps ---
    df_train_locations = df_features.iloc[:split_index].copy()
    
    # Calculate Outcome
    df_test_performance.loc[:, 'Outcome'] = np.where(
        (df_test_performance['TARGET_HIGH_RISK'] == 1) & (df_test_performance['Predicted_Risk'] == 1), 'TP',
        np.where(
            (df_test_performance['TARGET_HIGH_RISK'] == 0) & (df_test_performance['Predicted_Risk'] == 1), 'FP',
            np.where(
                (df_test_performance['TARGET_HIGH_RISK'] == 1) & (df_test_performance['Predicted_Risk'] == 0), 'FN',
                'TN'
            )
        )
    )
    map_center = [df_features['Lat'].mean(), df_features['Lon'].mean()]

    # --- MAP 1: Train/Test Split Sample (Sampling TIME-STEPS) ---
    
    # Calculate sizes based on n_samples (80/20 split)
    train_sample_size_m1 = int(n_samples * 0.8)
    test_sample_size_m1 = n_samples - train_sample_size_m1
    
    np.random.seed(42)
    
    # Sample training points (time-steps)
    df_train_sample = df_train_locations.sample(
        n=min(len(df_train_locations), train_sample_size_m1), replace=False
    )[['Lat', 'Lon']]

    # Sample test points (time-steps) directly from the performance DataFrame
    df_test_sample_performance = df_test_performance.sample(
        n=min(len(df_test_performance), test_sample_size_m1), replace=False
    )
    
    df_test_sample_m1 = df_test_sample_performance[['Lat', 'Lon']]
    
    actual_train_sample_size = len(df_train_sample)
    actual_test_sample_size = len(df_test_sample_m1) # This is the N for Map 2
    
    m1 = folium.Map(location=map_center, zoom_start=11, tiles='cartodbpositron')
    
    # Add data points
    for _, row in df_train_sample.iterrows():
        folium.Circle(location=[row['Lat'], row['Lon']], radius=5, color='blue', fill=True, fill_color='blue', fill_opacity=0.6, tooltip="Train Point").add_to(m1)
    for _, row in df_test_sample_m1.iterrows():
        folium.Circle(location=[row['Lat'], row['Lon']], radius=5, color='red', fill=True, fill_color='red', fill_opacity=0.7, tooltip="Test Point").add_to(m1)
    
    # Update Map 1 Legend text
    legend_html_m1 = f'''
        <div style="position: fixed; bottom: 50px; left: 50px; width: 170px; height: 100px; border:2px solid grey; z-index:9999; font-size:12px; background-color: white; opacity: 0.9;">
          &nbsp; <b>Map 1: Data Split (80/20)</b> <br>
          &nbsp; <i style="color:blue" class="fa fa-circle fa-1x"></i> Training Data ({actual_train_sample_size} points) <br>
          &nbsp; <i style="color:red" class="fa fa-circle fa-1x"></i> Test Data ({actual_test_sample_size} points) <br>
        </div>
        '''
    m1.get_root().html.add_child(folium.Element(legend_html_m1))
    
    # --- MAP 2: Performance Check (Non-TN Outcomes) ---
    
    # Filter to exclude True Negatives ONLY from the sampled test data.
    df_map2_final_sample = df_test_sample_performance[df_test_sample_performance['Outcome'] != 'TN'].copy()
    
    # Get counts for title/legend
    tp_count = len(df_map2_final_sample[df_map2_final_sample['Outcome'] == 'TP'])
    fp_count = len(df_map2_final_sample[df_map2_final_sample['Outcome'] == 'FP'])
    fn_count = len(df_map2_final_sample[df_map2_final_sample['Outcome'] == 'FN'])
    total_performance_points_m2 = tp_count + fp_count + fn_count # The number of points actually plotted

    # Custom Cluster Icon Logic (remains the same)
    js_icon_create_function = """
        function(cluster) {
            var markers = cluster.getAllChildMarkers();
            var counts = {'TP': 0, 'FP': 0, 'FN': 0};
            var dominantColor = 'green'; // Default color

            // Count the outcomes within the cluster
            markers.forEach(function(marker) {
                var outcome = marker.options.icon.options.html.match(/data-outcome="([^"]*)"/);
                if (outcome) {
                    counts[outcome[1]]++;
                }
            });

            var total = counts.TP + counts.FP + counts.FN;
            
            // Define Ambiguity Tolerance (e.g., within 10% of each other, relative to total)
            var tolerance = 0.10; 
            
            // --- 1. Check for TP/FP Ambiguity OR FN Dominance (Orange) ---
            if (Math.abs(counts.TP - counts.FP) / total < tolerance && total > 0 && (counts.TP + counts.FP > counts.FN)) {
                dominantColor = 'orange'; // Orange for Ambiguity
            } 
            else if (counts.FN > counts.TP && counts.FN > counts.FP) {
                dominantColor = 'orange'; // Orange for FN dominance (Simplified)
            }
            // --- 2. Check for FP Dominance (Red) ---
            else if (counts.FP > counts.TP && counts.FP > counts.FN) {
                dominantColor = 'red';
            } 
            // --- 3. Default to TP Dominance (Green) ---
            else { 
                dominantColor = 'green';
            }
            
            // Return the custom L.DivIcon with the dominant color
            return L.divIcon({ 
                html: '<div style="background-color:' + dominantColor + '; color: white; border-radius: 50%; width: 40px; height: 40px; line-height: 40px; text-align: center; font-weight: bold;">' + cluster.getChildCount() + '</div>',
                className: 'marker-cluster',
                iconSize: new L.Point(40, 40)
            });
        }
    """

    m2 = folium.Map(location=map_center, zoom_start=11, tiles='cartodbpositron')
    dominance_cluster = MarkerCluster(icon_create_function=js_icon_create_function).add_to(m2)
    color_map = {'TP': 'green', 'FP': 'red', 'FN': 'orange'}
    icon_map = {'TP': 'check', 'FP': 'times', 'FN': 'exclamation-triangle'}

    for _, row in df_map2_final_sample.iterrows():
        outcome = row['Outcome']
        color = color_map.get(outcome)
        icon_type = icon_map.get(outcome)
        icon_html = f"""<div style="text-align: center;" data-outcome="{outcome}"><i class="fa fa-{icon_type} fa-2x" style="color:{color}"></i></div>"""
        icon = folium.DivIcon(html=icon_html, icon_size=(24, 24))
        folium.Marker(location=[row['Lat'], row['Lon']], icon=icon, tooltip=f"Outcome: {outcome}").add_to(dominance_cluster)
        
    # Add Map 2 Legends (remains the same)
    legend_icon_key = f'''
        <div style="position: fixed; bottom: 50px; left: 50px; width: 220px; height: 100px; border:2px solid grey; z-index:9999; font-size:12px; background-color: white; opacity: 0.9;">
          &nbsp; <b>MARKER ICON KEY</b> <br>
          &nbsp; <i style="color:green" class="fa fa-check fa-1x"></i> True Positive (Successful Risk Identification) <br>
          &nbsp; <i style="color:red" class="fa fa-times fa-1x"></i> False Positive (Wasted Inspection) <br>
          &nbsp; <i style="color:orange" class="fa fa-exclamation-triangle fa-1x"></i> False Negative (Missed Risk) <br>
        </div>
        '''
    m2.get_root().html.add_child(folium.Element(legend_icon_key))

    legend_cluster_key = f'''
        <div style="position: fixed; bottom: 50px; right: 20px; width: 140px; height: 100px; border:2px solid grey; z-index:9999; font-size:12px; background-color: white; opacity: 0.9;">
          &nbsp; <b>CLUSTER COLORS</b> <br>
          &nbsp; <i style="color:green" class="fa fa-circle fa-1x"></i> TP Dominance <br>
          &nbsp; <i style="color:red" class="fa fa-circle fa-1x"></i> FP Dominance <br>
          &nbsp; <i style="color:orange" class="fa fa-circle fa-1x"></i> Mixed Zone <br>
        </div>
        '''
    m2.get_root().html.add_child(folium.Element(legend_cluster_key))


    # Combine maps into a single HTML structure for display/saving
    m1_html = m1._repr_html_()
    m2_html = m2._repr_html_()
    
    # --- Corrected Titles reflecting time-step sampling ---
    map1_title = f"Map 1: Chronological Train/Test Split (80/20 split)"
    map1_subtitle = f"Plotted: N={n_samples} Samples"
    map2_title = f"Map 2: Model Performance on Test Set"
    map2_subtitle = f"Plotted: {total_performance_points_m2} Non-True Negative Outcomes from the Test Set"


    final_html = f"""
    <div style="display: flex; justify-content: space-around; width: 100%;">
        <div style="width: 50%; padding-right: 5px;">
            <h3>{map1_title}</h3>
            <p style="margin-top: -15px; margin-bottom: 5px; font-size: 14px; color: #555;">{map1_subtitle}</p>
            {m1_html}
        </div>
        <div style="width: 50%; padding-left: 5px;">
            <h3>{map2_title}</h3>
            <p style="margin-top: -15px; margin-bottom: 5px; font-size: 14px; color: #555;">{map2_subtitle}</p>
            {m2_html}
        </div>
    </div>
    """
    
    # Save the dual map HTML to the reports directory
    with open(os.path.join(REPORT_DIR, 'dual_train_test_map.html'), 'w') as f:
        f.write(final_html)
    print(f"Saved Dual Train/Test/Performance Map to {REPORT_DIR}/dual_train_test_map.html")


if __name__ == '__main__':
    start_time = time.time()
    
    # 1. Load Artifacts
    artifacts = load_artifacts()
    if artifacts is None:
        exit()
        
    # 2. Generate Evaluation Artifacts
    generate_eda_charts(artifacts['df_features'])
    generate_feature_importance(artifacts['best_model'])
    generate_dual_map(artifacts)
    
    end_time = time.time()
    print(f"\n--- Evaluation Complete ---")
    print(f"Total evaluation time: {end_time - start_time:.2f} seconds.")