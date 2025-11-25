import pandas as pd
import numpy as np
import os
import time
import pickle
import warnings
import json 
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from sklearn.base import clone
from itertools import product
from tqdm.auto import tqdm 

# --- Path Setup to handle execution from inside 'src' ---
# Get the absolute path to the directory containing this script (e.g., /path/to/project/src)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# The project root is one directory up (e.g., /path/to/project)
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# --- Directory Configuration ---
# Root artifacts directory (for models, reports, splits)
ARTIFACTS_DIR_ROOT = os.path.join(PROJECT_ROOT, 'artifacts') 
# Input data directory (Assuming df_features_final.csv is saved here by data_pipeline.py)
DATA_PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data')

# Organized Artifact Subdirectories
MODEL_DIR = os.path.join(ARTIFACTS_DIR_ROOT, 'models')
REPORT_DIR = os.path.join(ARTIFACTS_DIR_ROOT, 'reports')
SPLIT_DIR = os.path.join(ARTIFACTS_DIR_ROOT, 'data_splits')

# Create necessary directories
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(SPLIT_DIR, exist_ok=True)

# Define Features (Must match data_pipeline.py)
COUNT_FEATURES = ['lag_1_rat', 'lag_1_dumping', 'lag_2_rat', 'lag_6_rat', 'lag_3_avg_rat']
TIME_FEATURES = ['month', 'year']
FEATURES = COUNT_FEATURES + TIME_FEATURES
TARGET = 'TARGET_HIGH_RISK'

# --- File Paths ---
FEATURES_FILE_NAME = 'df_features_final.csv'
# FIX APPLIED HERE: Path now points to the 'data' directory for the INPUT file
FEATURES_FILE_PATH = os.path.join(DATA_PROCESSED_DIR, FEATURES_FILE_NAME) 

# Paths for outputs in the organized artifacts folders
MODEL_PATH = os.path.join(MODEL_DIR, 'best_model.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')
METRICS_PATH = os.path.join(REPORT_DIR, 'model_metrics.json')
TEST_DATA_OUTPUT_PATH = os.path.join(SPLIT_DIR, 'X_test_features.csv')
SPLIT_INDEX_PATH = os.path.join(SPLIT_DIR, 'split_index.txt')

print(f"Artifacts will be saved in: {ARTIFACTS_DIR_ROOT}/{{models, reports, data_splits}}")

# --- WARNING SUPPRESSION ---
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# ====================================================================
# --- Custom GridSearchCV with TQDM Progress Bar ---
# ====================================================================
class TqdmGridSearchCV(GridSearchCV):
    """Custom GridSearchCV that provides a tqdm progress bar for parameter sets."""
    def fit(self, X, y=None, **kwargs):
        # Calculate total parameter sets 
        param_combinations = [dict(zip(self.param_grid.keys(), v)) 
                              for v in product(*self.param_grid.values())]
        total_iters = len(param_combinations)
        
        with tqdm(total=total_iters, desc="GridSearch Progress (Param Sets)") as pbar:
            best_score_ = -np.inf
            
            # Note: The manual iteration here is for visualization only. 
            # The super().fit() call performs the actual scoring and selection.
            for i, params in enumerate(param_combinations):
                # Using dummy update to show progress based on number of combinations
                pbar.update(1)
                pbar.set_postfix({'Combinations Checked': f'{i+1}/{total_iters}'})
                
            # Run the final, full GridSearchCV.fit 
            # We reset the progress bar for the actual fitting process if verbose > 0
            pbar.close() # Close the combination counting bar
            
            # The original GridSearchCV implementation handles the cross-validation
            super().fit(X, y, **kwargs)

        return self

# ====================================================================
# --- Metrics Saving Function ---
# ====================================================================
def save_model_metrics(model, X_test, y_test, features):
    """
    Calculates key metrics, feature importance, and saves them to a JSON file 
    for the dashboard to consume dynamically.
    """
    
    # Generate predictions and probabilities
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate performance metrics
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    auc = roc_auc_score(y_test, y_prob)

    # Calculate Feature Importance
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': features,
        'score': importances
    }).sort_values(by='score', ascending=False)
    
    # Get top 5 features with their scores and assign descriptive names
    top_features = feature_importance_df.head(5).to_dict('records')
    # Enhance feature descriptions based on our pipeline context
    for f in top_features:
        if 'lag_3_avg_rat' in f['feature']:
            f['description'] = "Chronic Rat Issues (3-month average of sightings)"
        elif 'lag_1_dumping' in f['feature']:
            f['description'] = "Neglect Signal (Prior 1-month illegal dumping incidents)"
        elif 'lag_1_rat' in f['feature']:
            f['description'] = "Recent Rat Activity (Prior 1-month count of sightings)"
        elif '311_total' in f['feature']:
            f['description'] = "Total 311 service requests (proxy for neighborhood activity)"
        else:
            f['description'] = f['feature'] # Fallback (no specific description)
            
    # Structure the final data
    metrics_data = {
        # High-Risk Class (1) is the positive class we want to prioritize
        "AUC": round(auc, 3),
        "class_1_recall": round(report['1']['recall'], 3),
        "class_1_precision": round(report['1']['precision'], 3),
        "class_1_f1": round(report['1']['f1-score'], 3),
        "top_features": top_features
    }

    # Save to JSON file in the reports directory
    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics_data, f, indent=4)
        
    print(f"Metrics saved to {METRICS_PATH}")
    
    return metrics_data

# ====================================================================
# --- Main Training Function ---
# ====================================================================
def train_model():
    """Loads data, prepares ML inputs, trains and evaluates the model, and saves artifacts."""
    print("--- 1. Loading Features Data ---")
    try:
        # Load data from the specified path (now pointing to the 'data' directory)
        df_features = pd.read_csv(FEATURES_FILE_PATH)
    except FileNotFoundError:
        # Update error message to reflect the corrected path
        print(f"Error: Features file not found at {FEATURES_FILE_PATH}. Please ensure data_pipeline.py has run and saved to the '{os.path.basename(DATA_PROCESSED_DIR)}' directory.")
        return

    print(f"Loaded {len(df_features)} block-month records for training.")

    # 2. ML Model Prep and Splitting
    print("\n--- 2. ML Model Prep and Sequential Splitting ---")
    
    # Sort the dataframe chronologically to ensure a true time-series split
    df_features = df_features.sort_values(by=['year', 'month', 'Block_ID'], ascending=True).reset_index(drop=True)

    X = df_features[FEATURES].copy()
    y = df_features[TARGET]

    # Determine the chronological split index (80% for training)
    split_index = int(len(X) * 0.8)

    # Split X and y sequentially (Time-series validation)
    X_train = X.iloc[:split_index].copy()
    X_test = X.iloc[split_index:].copy()
    y_train = y.iloc[:split_index] 
    y_test = y.iloc[split_index:] 
    
    # Save the split index for later map visualization in model_evaluate.py
    with open(SPLIT_INDEX_PATH, 'w') as f:
        f.write(str(split_index))

    # 3. Selective Scaling (StandardScaler)
    scaler = StandardScaler()
    # Only scale the count-based features, leaving month/year as is
    X_train[COUNT_FEATURES] = scaler.fit_transform(X_train[COUNT_FEATURES])
    X_test[COUNT_FEATURES] = scaler.transform(X_test[COUNT_FEATURES])
    print(f"Data Split: Train size={len(X_train)}, Test size={len(X_test)}")

    # 4. Hyperparameter Tuning with TQDM
    print("\n--- 3. Starting Hyperparameter Tuning (Random Forest) ---")
    start_time = time.time()

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_leaf': [1, 5],
        'class_weight': ['balanced', {0: 1, 1: 5}] # Address class imbalance
    }

    rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)

    grid_search = TqdmGridSearchCV(
        estimator=rf_base,
        param_grid=param_grid,
        # F1 is often preferred for imbalanced classification
        scoring='f1',
        cv=3,
        verbose=0
    )

    # We use the scaled training data for tuning
    grid_search.fit(X_train, y_train) 
    best_model = grid_search.best_estimator_
    end_time = time.time()

    print(f"\nGridSearchCV complete in {end_time - start_time:.2f} seconds.")
    print(f"Best Model Parameters: {grid_search.best_params_}")

    # 5. Model Evaluation (on Test Set)
    print("\n--- 4. Model Evaluation on Test Set ---")
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1] # Needed for AUC

    print(f"Overall Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(f"AUC Score: {roc_auc_score(y_test, y_prob):.3f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # 6. Save Model and Scaler Artifacts (to models directory)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(best_model, f)
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    
    # 7. Save Test Set for Evaluation Visualization (to data_splits directory)
    # Merge the required identification columns (Block_ID, Month_Year) back into X_test_output
    df_id_test = df_features.iloc[split_index:][['Block_ID', 'Month_Year']].reset_index(drop=True)
    
    X_test_output = X_test.copy().reset_index(drop=True)
    X_test_output[['Block_ID', 'Month_Year']] = df_id_test 
    
    # Add target and predictions
    X_test_output['TARGET_HIGH_RISK'] = y_test.values
    X_test_output['Predicted_Risk'] = y_pred
    
    X_test_output.to_csv(TEST_DATA_OUTPUT_PATH, index=False)
    
    # 8. Save Metrics for Streamlit Dashboard (to reports directory)
    save_model_metrics(best_model, X_test, y_test, FEATURES)
    
    print(f"\nSaved best model to {MODEL_PATH}")
    print(f"Saved scaler to {SCALER_PATH}")
    print(f"Saved test data to {TEST_DATA_OUTPUT_PATH}")
    print(f"Saved split index to {SPLIT_INDEX_PATH}")

if __name__ == '__main__':
    train_model()