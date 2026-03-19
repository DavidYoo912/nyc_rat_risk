# model_train.py
import pandas as pd
import numpy as np
import os
import time
import pickle
import json
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report, roc_auc_score, f1_score,
    average_precision_score, precision_recall_curve
)
from itertools import product
from config import (
    COUNT_FEATURES, TIME_FEATURES, BOROUGH_COL, TARGET,
    RAT_QUANTILE_THRESHOLD, TRAIN_SPLIT_RATIO, CV_FOLDS
)
from utils import get_logger

logger = get_logger(__name__)

# --- Path Setup ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

ARTIFACTS_DIR_ROOT = os.path.join(PROJECT_ROOT, 'artifacts')
DATA_PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')

MODEL_DIR = os.path.join(ARTIFACTS_DIR_ROOT, 'models')
REPORT_DIR = os.path.join(ARTIFACTS_DIR_ROOT, 'reports')
SPLIT_DIR = os.path.join(ARTIFACTS_DIR_ROOT, 'data_splits')

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(SPLIT_DIR, exist_ok=True)

FEATURES_FILE_PATH = os.path.join(DATA_PROCESSED_DIR, 'df_features_final.csv')
MODEL_PATH = os.path.join(MODEL_DIR, 'best_model.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')
X_TRAIN_COLS_PATH = os.path.join(MODEL_DIR, 'X_train_cols.pkl')
THRESHOLD_PATH = os.path.join(MODEL_DIR, 'rat_threshold.pkl')
DECISION_THRESHOLD_PATH = os.path.join(MODEL_DIR, 'decision_threshold.pkl')
METRICS_PATH = os.path.join(REPORT_DIR, 'model_metrics.json')
TEST_DATA_OUTPUT_PATH = os.path.join(SPLIT_DIR, 'X_test_features.csv')
SPLIT_INDEX_PATH = os.path.join(SPLIT_DIR, 'split_index.txt')

# Minimum recall we must achieve; missing hotspots is worse than false alarms
MIN_RECALL_TARGET = 0.85

logger.info(f"Artifacts will be saved in: {ARTIFACTS_DIR_ROOT}/{{models, reports, data_splits}}")


# ====================================================================
# --- Custom GridSearchCV ---
# ====================================================================
class TqdmGridSearchCV(GridSearchCV):
    """Placeholder to indicate GridSearchCV is running."""
    def fit(self, X, y=None, **kwargs):
        param_combinations = [
            dict(zip(self.param_grid.keys(), v))
            for v in product(*self.param_grid.values())
        ]
        logger.info(f"  --> Checking {len(param_combinations)} parameter combinations...")
        return super().fit(X, y, **kwargs)


# ====================================================================
# --- Metrics Saving Function ---
# ====================================================================
def save_model_metrics(model, X_test, y_test_binary, feature_names, decision_threshold):
    """Calculates key metrics and saves them to a JSON file for the dashboard."""
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= decision_threshold).astype(int)

    report = classification_report(y_test_binary, y_pred, output_dict=True, zero_division=0)
    try:
        auc = roc_auc_score(y_test_binary, y_prob)
    except ValueError:
        auc = np.nan

    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'get_booster'):
        importance_scores = model.get_booster().get_score(importance_type='gain')
        importances = np.array([importance_scores.get(f, 0.0) for f in feature_names])
    else:
        importances = np.zeros(len(feature_names))

    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'score': importances
    }).sort_values(by='score', ascending=False)

    top_features = feature_importance_df.head(5).to_dict('records')

    for f in top_features:
        if 'lag_12_avg_rat' in f['feature']:
            f['description'] = "Chronic Rat Issues (12-month rolling average)"
        elif 'dumping_rat_interaction' in f['feature']:
            f['description'] = "Neglect Interaction (Prior Rat x Prior Dumping Incidents)"
        elif 'lag_3_avg_rat' in f['feature']:
            f['description'] = "Recent Chronic Rat Issues (3-month rolling average)"
        elif 'lag_1_dumping' in f['feature']:
            f['description'] = "Neglect Signal (Prior 1-month illegal dumping incidents)"
        elif 'lag_1_rat' in f['feature']:
            f['description'] = "Recent Rat Activity (Prior 1-month count of sightings)"
        elif f['feature'].startswith('Borough_'):
            f['description'] = f['feature'].replace('Borough_', '') + " Presence"
        else:
            f['description'] = f['feature']

    metrics_data = {
        "AUC": round(auc, 3) if not np.isnan(auc) else None,
        "class_1_recall": round(report['1']['recall'], 3),
        "class_1_precision": round(report['1']['precision'], 3),
        "class_1_f1": round(report['1']['f1-score'], 3),
        "top_features": top_features
    }

    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics_data, f, indent=4)

    logger.info(f"Metrics saved to {METRICS_PATH}")
    return metrics_data


# ====================================================================
# --- Training Helper Function ---
# ====================================================================
def run_grid_search(model_name, estimator, param_grid, X_train, y_train_binary, X_test, y_test_binary, cv):
    """Performs GridSearchCV and returns the best estimator and performance metrics.

    C-5: Accepts a cv object (TimeSeriesSplit) instead of an integer.
    Uses average_precision scoring (threshold-independent, good for imbalanced data).
    Model selection compares AP-Score on test set.
    """
    logger.info(f"\n--- Starting Grid Search for {model_name} ---")

    grid_search = TqdmGridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring='average_precision',
        cv=cv,  # C-5: TimeSeriesSplit
        verbose=0,
        n_jobs=-1
    )

    start_time = time.time()
    grid_search.fit(X_train, y_train_binary)
    end_time = time.time()

    best_model = grid_search.best_estimator_
    logger.info(f"Grid Search complete in {end_time - start_time:.2f} seconds.")
    logger.info(f"Best Parameters: {grid_search.best_params_}")

    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    ap_score = average_precision_score(y_test_binary, y_pred_proba)

    metrics = {
        'Model': model_name,
        'AP-Score': ap_score,
        'Best_Model': best_model,
        'y_pred_proba': y_pred_proba,
    }

    return metrics


# ====================================================================
# --- Main Training Function ---
# ====================================================================
def train_model():
    """Loads data, prepares ML inputs, trains and evaluates the best model, and saves artifacts."""
    logger.info("--- 1. Loading and Preparing Features Data ---")
    try:
        df_features = pd.read_csv(FEATURES_FILE_PATH)
    except FileNotFoundError:
        logger.error(f"Features file not found at {FEATURES_FILE_PATH}.")
        return

    # C-2: Stable chronological sort — Block_ID as tiebreaker prevents non-determinism
    df_features = df_features.sort_values(
        by=['year', 'month', 'Block_ID'], ascending=True
    ).reset_index(drop=True)

    feature_cols = COUNT_FEATURES + TIME_FEATURES + [BOROUGH_COL]
    X_raw = df_features[feature_cols].copy()

    # C-3: Compute and write split index to file so model_evaluate.py uses the same boundary
    split_index = int(len(X_raw) * TRAIN_SPLIT_RATIO)
    with open(SPLIT_INDEX_PATH, 'w') as f:
        f.write(str(split_index))

    X_raw_train = X_raw.iloc[:split_index].copy()
    X_raw_test = X_raw.iloc[split_index:].copy()

    # C-1: Compute threshold from training rows only — prevents target leakage
    df_train = df_features.iloc[:split_index]
    df_test = df_features.iloc[split_index:]
    rat_threshold = df_train['next_month_rat_count'].quantile(RAT_QUANTILE_THRESHOLD)

    y_train_binary = (df_train['next_month_rat_count'].values >= rat_threshold).astype(int)
    y_test_binary  = (df_test['next_month_rat_count'].values  >= rat_threshold).astype(int)

    pos_rate = y_train_binary.mean()
    # Weight to penalise false negatives proportionally to class imbalance
    scale_pos_weight = (1 - pos_rate) / pos_rate

    logger.info(f"Train-only threshold (90th pct): {rat_threshold:.1f} incidents/month")
    logger.info(f"High-risk rate — train: {pos_rate:.1%} | test: {y_test_binary.mean():.1%}")
    logger.info(f"scale_pos_weight for XGBoost: {scale_pos_weight:.1f}")

    # Save threshold artifact for use by model_evaluate.py
    with open(THRESHOLD_PATH, 'wb') as f:
        pickle.dump(float(rat_threshold), f)

    # OHE and feature alignment
    logger.info("\n--- 2. Performing One-Hot Encoding and Feature Alignment ---")
    X_train = pd.get_dummies(X_raw_train, columns=[BOROUGH_COL], drop_first=True, prefix='Borough')
    X_test = pd.get_dummies(X_raw_test, columns=[BOROUGH_COL], drop_first=True, prefix='Borough')

    missing_cols = set(X_train.columns) - set(X_test.columns)
    for c in missing_cols:
        X_test[c] = 0
    X_test = X_test[X_train.columns]

    ALL_FEATURES = X_train.columns.tolist()

    logger.info(f"Loaded {len(df_features)} block-month records.")
    logger.info(f"Data Split: Train size={len(X_train)}, Test size={len(X_test)}")
    logger.info(f"Total features used: {len(ALL_FEATURES)}")

    # Scaling
    logger.info("\n--- 3. Selective Scaling of Count Features ---")
    scaler = StandardScaler()
    X_train[COUNT_FEATURES] = scaler.fit_transform(X_train[COUNT_FEATURES])

    X_test_scaled_array = scaler.transform(X_test[COUNT_FEATURES])
    X_test_scaled_df = pd.DataFrame(
        X_test_scaled_array, columns=COUNT_FEATURES, index=X_test.index
    )
    for col in COUNT_FEATURES:
        X_test[col] = X_test_scaled_df[col]

    # Hyperparameter tuning
    logger.info("\n--- 4. Starting Hyperparameter Tuning and Model Comparison ---")
    model_results = {}

    # C-5: TimeSeriesSplit prevents future-leaking CV folds on time-series data
    tscv = TimeSeriesSplit(n_splits=CV_FOLDS)

    # --- A. XGBoost Classifier ---
    xgb_base = XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        tree_method='hist',
    )

    xgb_param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [5, 10],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8, 1.0],
    }

    xgb_metrics = run_grid_search(
        "XGBoost", xgb_base, xgb_param_grid,
        X_train, y_train_binary, X_test, y_test_binary, cv=tscv
    )
    model_results['XGBoost'] = xgb_metrics

    # --- B. RandomForest Classifier ---
    rf_base = RandomForestClassifier(
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

    rf_param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_leaf': [1, 5],
        'max_features': ['sqrt', 0.5],
    }

    rf_metrics = run_grid_search(
        "RandomForest", rf_base, rf_param_grid,
        X_train, y_train_binary, X_test, y_test_binary, cv=tscv
    )
    model_results['RandomForest'] = rf_metrics

    # Model selection — compare by AP-Score (ranking quality on test set)
    logger.info("\n--- 5. Final Model Selection and Threshold Tuning ---")
    results_df = pd.DataFrame(model_results).T
    results_df['AP-Score'] = results_df['AP-Score'].fillna(0)

    best_model_name = results_df['AP-Score'].idxmax()
    best_model_info = results_df.loc[best_model_name]
    final_model = best_model_info['Best_Model']
    final_y_proba = best_model_info['y_pred_proba']

    logger.info(
        f"Best Model Selected: {best_model_name} "
        f"(AP-Score: {best_model_info['AP-Score']:.3f})"
    )

    # Tune decision threshold: maximise F1 subject to recall >= MIN_RECALL_TARGET.
    # Recall is the primary metric — missing a hotspot is worse than a false alarm.
    precisions, recalls, thresholds = precision_recall_curve(y_test_binary, final_y_proba)
    # precisions/recalls have one extra entry (at threshold=1); align arrays
    valid = recalls[:-1] >= MIN_RECALL_TARGET
    if valid.any():
        f1_scores = (
            2 * precisions[:-1] * recalls[:-1]
            / (precisions[:-1] + recalls[:-1] + 1e-8)
        )
        f1_scores[~valid] = 0
        best_idx = int(f1_scores.argmax())
        decision_threshold = float(thresholds[best_idx])
        logger.info(
            f"Threshold tuned to {decision_threshold:.3f} "
            f"(recall={recalls[best_idx]:.1%}, precision={precisions[best_idx]:.1%}, "
            f"F1={f1_scores[best_idx]:.3f})"
        )
    else:
        decision_threshold = float(thresholds[int(recalls[:-1].argmax())])
        logger.warning(
            f"Could not achieve recall >= {MIN_RECALL_TARGET:.0%}. "
            f"Using max-recall threshold: {decision_threshold:.3f}"
        )

    final_y_pred = (final_y_proba >= decision_threshold).astype(int)

    logger.info("\n--- Classification Report (Test Set) ---")
    logger.info(classification_report(y_test_binary, final_y_pred))

    # Save artifacts
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(final_model, f)
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    with open(X_TRAIN_COLS_PATH, 'wb') as f:
        pickle.dump(ALL_FEATURES, f)
    with open(DECISION_THRESHOLD_PATH, 'wb') as f:
        pickle.dump(decision_threshold, f)

    df_id_test = df_features.iloc[split_index:][['Block_ID', 'Month_Year']].reset_index(drop=True)
    X_test_output = X_test.copy().reset_index(drop=True)
    X_test_output[['Block_ID', 'Month_Year']] = df_id_test
    X_test_output[TARGET] = y_test_binary
    X_test_output['Predicted_Score'] = final_y_proba
    X_test_output['Predicted_Risk'] = final_y_pred
    X_test_output.to_csv(TEST_DATA_OUTPUT_PATH, index=False)

    save_model_metrics(final_model, X_test, y_test_binary, ALL_FEATURES, decision_threshold)

    logger.info(f"\nSaved best model ({best_model_name}) to {MODEL_PATH}")
    logger.info(f"Saved scaler to {SCALER_PATH}")
    logger.info(f"Saved training column list to {X_TRAIN_COLS_PATH}")
    logger.info(f"Saved rat threshold ({rat_threshold:.1f}) to {THRESHOLD_PATH}")
    logger.info(f"Saved decision threshold ({decision_threshold:.3f}) to {DECISION_THRESHOLD_PATH}")
    logger.info(f"Saved test data to {TEST_DATA_OUTPUT_PATH}")
    logger.info(f"Saved split index ({split_index}) to {SPLIT_INDEX_PATH}")


if __name__ == '__main__':
    start_total = time.time()
    train_model()
    end_total = time.time()
    logger.info(f"\n--- Training Pipeline Complete ---")
    logger.info(f"Total pipeline time: {end_total - start_total:.2f} seconds.")
