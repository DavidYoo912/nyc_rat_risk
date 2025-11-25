import streamlit as st
import os
import pandas as pd
import json
from datetime import datetime
from dateutil.relativedelta import relativedelta

# --- Path Setup to handle execution location ---
# Get the absolute path to the directory containing this script (assumed to be project root)
APP_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Configuration ---
# Base artifacts directory
ARTIFACTS_DIR = os.path.join(APP_DIR, 'artifacts')
# NEW: Subdirectory for reports and outputs (based on user's directory image)
REPORTS_DIR = os.path.join(ARTIFACTS_DIR, 'reports')

# Define paths to the generated artifacts using the REPORTS_DIR (FIXED PATHS)
PATHS = {
    # Landing Page Artifact (Now correctly pointing to 'artifacts/reports/')
    'prediction_map': os.path.join(REPORTS_DIR, 'risk_map_deployment.html'),
    # Project Details Artifacts
    'eda_chart': os.path.join(REPORTS_DIR, 'eda_charts.png'),
    'eval_map': os.path.join(REPORTS_DIR, 'dual_train_test_map.html'),
    'feature_importance_chart': os.path.join(REPORTS_DIR, 'feature_importance.png'),
    'metrics_data': os.path.join(REPORTS_DIR, 'model_metrics.json'),
    # Data for Download (Find the latest inspection list)
    'inspection_list_prefix': 'inspection_list_'
}

def load_html_content(filepath):
    """Loads and returns the HTML content of a file."""
    try:
        with open(filepath, 'r') as f:
            return f.read()
    except FileNotFoundError:
        st.error(f"Artifact not found: {os.path.basename(filepath)}. Please ensure you have run the model pipeline successfully.")
        return None

def load_model_metrics():
    """
    Loads metrics and feature importance data from the JSON artifact.
    Returns an empty dict if the file is not found.
    """
    metrics_path = PATHS['metrics_data']
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            st.warning(f"Error decoding JSON from {metrics_path}. Metrics will not be displayed.")
            return {}
    else:
        # Updated warning message to show the correct REPORTS_DIR
        st.warning(f"Metrics file ({os.path.basename(metrics_path)}) not found in {os.path.basename(REPORTS_DIR)}. Please run the model pipeline to generate it.")
        return {}

def find_latest_inspection_list(prefix):
    """Finds the most recently created CSV file starting with the prefix."""
    try:
        # Check if the absolute reports directory exists (FIXED DIRECTORY CHECK)
        if not os.path.exists(REPORTS_DIR):
            return None, None
            
        # List files in the absolute reports directory
        files = [f for f in os.listdir(REPORTS_DIR) if f.startswith(prefix) and f.endswith('.csv')]
        if not files:
            return None, None
        
        # Sort by modification time (most recent first) using absolute paths
        files.sort(key=lambda x: os.path.getmtime(os.path.join(REPORTS_DIR, x)), reverse=True)
        # Use REPORTS_DIR to join path
        latest_file = os.path.join(REPORTS_DIR, files[0])
        
        # Extract the prediction month from the filename
        filename = os.path.basename(latest_file)
        parts = filename.replace('.csv', '').split('_')
        
        # Rejoin everything after 'inspection_list' to get the Month Year string
        # Expects filename format: 'inspection_list_MONTH_YEAR.csv'
        if len(parts) > 2:
            prediction_month_str = ' '.join(parts[2:]).upper()
        else:
            prediction_month_str = None
            
        return latest_file, prediction_month_str

    except Exception as e:
        st.error(f"Error finding inspection list: {e}")
        return None, None

def get_input_month_str(prediction_month_str):
    """Calculates the input data month (Prediction Month - 1)."""
    if not prediction_month_str:
        return "Unknown"
    
    try:
        # Parse "DECEMBER 2025" -> Date object
        pred_date = datetime.strptime(prediction_month_str, '%B %Y')
        # Subtract 1 month
        input_date = pred_date - relativedelta(months=1)
        # Format back to string
        return input_date.strftime('%B %Y').upper()
    except ValueError:
        return "Unknown"

def main():
    """Main Streamlit application function."""
    
    # --- Load Dynamic Data ---
    metrics = load_model_metrics()
    latest_csv_path, prediction_month_str = find_latest_inspection_list(PATHS['inspection_list_prefix'])
    
    # Calculate Dynamic Strings
    input_month_str = get_input_month_str(prediction_month_str) if prediction_month_str else "N/A"
    target_month_str = prediction_month_str if prediction_month_str else "Next Month"

    # --- Page Configuration ---
    # The theme setting is handled by .streamlit/config.toml
    st.set_page_config(
        page_title="NYC Rat Risk Predictor",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # =========================================================================
    # 📌 PERSISTENT SIDEBAR CONTENT (Project Background)
    # =========================================================================
    with st.sidebar:
        st.header("Project Background")
        st.markdown(
            """
            ### 🐀 Proactive Pest Mitigation
            This model uses historical data from **NYC Open Data** (311 Service Requests) to proactively identify 
            city blocks at the highest risk of severe rat infestations.
            
            **The Core Hypothesis:**
            Illegal dumping and garbage accumulation ('Neglect Signals') are leading indicators of rodent activity. 
            By detecting these signals early, the city can intervene *before* a rat colony becomes established.
            """
        )
        st.markdown("---")


    # =========================================================================
    # MAIN CONTENT AREA
    # =========================================================================
    st.title("🏙️ NYC Proactive Rat Risk Modeling")
    st.markdown("---")

    # --- Tabs Setup (Updated Structure) ---
    tab1, tab2, tab3 = st.tabs([
        "🚀 Live Deployment & Prediction", 
        "📘 Introduction & EDA",
        "📊 Model Evaluation"
    ])


    # =========================================================================
    # TAB 1: Live Deployment & Prediction
    # =========================================================================
    with tab1:
        st.header(f"Live Deployment Dashboard")
        
        # Dynamic Title showing the Current Data -> Next Month Prediction relationship
        st.markdown(f"### 🎯 Prediction Target: {target_month_str}")
        st.markdown(f"**Based on Input Data through: {input_month_str}**")
        
        prediction_map_html = load_html_content(PATHS['prediction_map'])
        
        if prediction_map_html:
            # Embed the Folium map HTML
            st.components.v1.html(
                prediction_map_html, 
                height=700,
                scrolling=False
            )
            
            # Download Section
            if latest_csv_path and os.path.exists(latest_csv_path):
                try:
                    df_inspection = pd.read_csv(latest_csv_path)
                    num_blocks = len(df_inspection)
                    
                    st.success(f"✅ Prediction Complete. Found **{num_blocks}** high-risk blocks.")
                    
                    col1, col2 = st.columns([1, 4])
                    with col1:
                        with open(latest_csv_path, "rb") as file:
                            st.download_button(
                                label="Download Inspection List (.csv)",
                                data=file,
                                file_name=os.path.basename(latest_csv_path),
                                mime="text/csv",
                                use_container_width=True
                            )
                    with col2:
                        st.info("Download this list to assign field inspection teams.")
                        
                except Exception as e:
                    st.warning(f"Could not load or read inspection list: {e}")
            else:
                st.warning("Prediction list CSV not found. Please run the model pipeline to generate the high-risk blocks list.")


    # =========================================================================
    # TAB 2: Introduction & Exploratory Data Analysis (EDA)
    # =========================================================================
    with tab2:
        st.header("Exploratory Data Analysis (EDA)")
        st.markdown(
            "The charts below validate the project hypothesis, showing the strong temporal and seasonal "
            "correlation between illegal dumping incidents and subsequent rat sightings."
        )
        
        if os.path.exists(PATHS['eda_chart']):
            st.image(PATHS['eda_chart'], caption="Analysis of Trends, Seasonality, and Neglect Correlations", use_container_width=True)
        else:
            st.error(f"Artifact not found: **{os.path.basename(PATHS['eda_chart'])}**. Please run the model pipeline.")


    # =========================================================================
    # TAB 3: Model Evaluation
    # =========================================================================
    with tab3:
        st.header("Model Performance Evaluation")
        
        # --- Section 1: Evaluation Map ---
        st.subheader("1. Spatial & Chronological Split Validation")
        st.markdown(
            """
            To prevent data leakage, the model was trained and tested using a strict **Chronological Split**:
            * **Blue Dots (Map 1):** Historical Training Data (80%)
            * **Red Dots (Map 1):** Recent Test Data (20%)
            
            **Map 2** highlights where the model succeeded (Green) and failed (Red/Orange) on the test set.
            """
        )
        
        eval_map_html = load_html_content(PATHS['eval_map'])
        
        if eval_map_html:
            st.components.v1.html(
                eval_map_html, 
                height=650,
                scrolling=False
            )

        st.markdown("---")
        
        # --- Section 2: Metrics (Dynamically Loaded) ---
        st.subheader("2. Key Performance Metrics")
        
        if metrics:
            st.markdown(
                """
                The model is optimized for **Recall** to minimize missed detections of high-risk blocks.
                The results below are based on the latest model evaluation on the chronological test set:
                """
            )
            
            # Display Metrics in columns
            col_auc, col_recall, col_precision, col_f1 = st.columns(4)
            
            col_auc.metric("AUC", f"~{metrics.get('AUC', 0.0):.2f}")
            col_recall.metric("Recall (High Risk)", f"~{metrics.get('class_1_recall', 0.0):.2f}")
            col_precision.metric("Precision (High Risk)", f"~{metrics.get('class_1_precision', 0.0):.2f}")
            col_f1.metric("F1-Score (High Risk)", f"~{metrics.get('class_1_f1', 0.0):.2f}")
        else:
            st.warning("Model metrics not loaded. Please ensure **model_metrics.json** is generated.")


        st.markdown("---")

        # --- Section 3: Feature Importance (Dynamically Loaded) ---
        st.subheader("3. Feature Importance")
        
        if metrics.get('top_features'):
            
            top_feature = metrics['top_features'][0]['feature']
            top_score = metrics['top_features'][0]['score']
            
            # Dynamic Description
            st.markdown(
                f"The model confirms that **Chronic Issues** (3-month average of rat sightings) and **Neglect Signals** "
                f"(prior dumping) are the strongest predictors of future risk. "
                f"The top-ranked feature is `{top_feature}` with a score of **{top_score:.2f}**."
            )
            
            st.markdown("#### Top Features influencing Prediction:")
            
            # Display feature importance table (using DataFrame for clean formatting)
            features_df = pd.DataFrame(metrics['top_features']).set_index('feature')
            st.dataframe(
                features_df[['score']].rename(columns={'score': 'Importance Score'}).head(5), 
                use_container_width=True
            )
            
        else:
            st.markdown(
                "Feature importance data not found in the metrics file. Run the model training pipeline to generate this data."
            )
            
        if os.path.exists(PATHS['feature_importance_chart']):
            st.image(PATHS['feature_importance_chart'], caption="Random Forest Feature Importance", use_container_width=True)
        else:
            st.error(f"Artifact not found: **{os.path.basename(PATHS['feature_importance_chart'])}**. Please run the model pipeline.")

if __name__ == '__main__':
    main()