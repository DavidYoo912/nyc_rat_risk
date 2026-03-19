# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NYC Rat Risk is a machine learning system that predicts high-risk city blocks for rat infestations using NYC 311 Service Request data. The core hypothesis: **Neglect Signals** (illegal dumping + garbage accumulation complaints) are leading indicators of rodent activity, enabling proactive intervention before colonies establish.

The system delivers an inspection list (top 200 high-risk blocks) and interactive risk map to city field teams via a Streamlit dashboard.

## Commands

### Run the full MLOps pipeline (data → train → evaluate → predict)
```bash
bash run_pipeline.sh
```
This creates/reuses `ml_env/` virtual environment, installs dependencies, and runs all 4 pipeline stages sequentially.

### Launch the Streamlit dashboard (requires pipeline artifacts to exist)
```bash
source ml_env/bin/activate
streamlit run app.py
```

### Run individual pipeline stages
```bash
source ml_env/bin/activate
python src/data_pipeline.py    # Fetches NYC API data, engineers features (~60s, makes network calls)
python src/model_train.py      # Trains XGBoost + RandomForest with GridSearchCV (~60s)
python src/model_evaluate.py   # Generates evaluation charts and maps (~30s)
python src/model_predict.py    # Generates live risk map + inspection CSV (~10s)
```

### Install dependencies
```bash
python3 -m venv ml_env
source ml_env/bin/activate
pip install -r requirements.txt
```

## Architecture

### Pipeline Flow

```
NYC 311 API (rat sightings + illegal dumping)
    ↓ data_pipeline.py
data/processed/df_features_final.csv  +  artifacts/rat_threshold.txt
    ↓ model_train.py
artifacts/models/{best_model.pkl, scaler.pkl, X_train_cols.pkl}  +  artifacts/reports/model_metrics.json
    ↓ model_evaluate.py
artifacts/reports/{eda_charts.png, feature_importance.*, dual_train_test_map.html, metrics_by_borough.csv}
    ↓ model_predict.py
artifacts/reports/{risk_map_deployment.html, inspection_list_[MONTH]_[YEAR].csv}
    ↓ app.py (Streamlit)
Interactive dashboard for city planners and field teams
```

### Feature Engineering (`data_pipeline.py`)

Geospatial binning: coordinates rounded to 4 decimal places (~11m precision) to define city "blocks" as `Block_ID`.

Key engineered features:
- `lag_1_rat`, `lag_2_rat`, `lag_6_rat` — lagged rat sighting counts (1, 2, 6 months prior)
- `lag_3_avg_rat`, `lag_12_avg_rat` — rolling averages (chronic risk patterns)
- `lag_1_dumping` — prior month illegal dumping count
- `dumping_rat_interaction` — `lag_1_rat × lag_1_dumping` (the core "neglect signal")
- `month`, `year` — temporal features
- Borough one-hot encoded at training time

**Target:** Binary — next month rat count ≥ 75th percentile threshold (saved to `artifacts/models/rat_threshold.pkl`)

### Model Training (`model_train.py`)

- **Chronological 80/20 split** (sorted by year/month) — critical to prevent temporal data leakage
- Compares XGBoost (typically wins) vs. RandomForest via GridSearchCV (F1-optimized, 3-fold CV)
- `X_train_cols.pkl` stores the exact feature column order — used to align test/prediction sets

### Live Prediction (`model_predict.py`)

Extracts the most recent month from `df_features_final.csv`, applies the same one-hot encoding and scaling, aligns to `X_train_cols.pkl` (filling missing borough columns with 0), and generates predictions.

### Streamlit App (`app.py`)

Three-tab layout:
- **Tab 1:** Live deployment map + downloadable inspection CSV (dynamically reads `inspection_list_*.csv`)
- **Tab 2:** EDA charts and project background
- **Tab 3:** Train/test evaluation maps, metrics from `model_metrics.json`, feature importance

All artifact paths are read dynamically from the `artifacts/` directory at runtime.

## Key Design Decisions

- **Chronological split only** — never use random split; it would leak future data into training
- **`X_train_cols.pkl` alignment** — prediction features must match training column order exactly; always use this file when generating live predictions
- **75th percentile threshold** — defines "high risk"; loaded from `artifacts/models/rat_threshold.pkl` at evaluation/prediction time. See reasoning below.
- **`dumping_rat_interaction`** must be recreated from raw lags before scaling, not derived from scaled values
- **Recall is the primary metric** — false negatives (missed high-risk blocks) are more costly than false positives
- **75th percentile threshold (not 90th)** — "High risk" means blocks with meaningfully elevated rat activity next month, not just extreme outliers. The 75th percentile produces a ~55% positive rate, which:
  - Gives the model enough positive examples to learn a meaningful decision boundary
  - Raises the AP-Score ceiling from ~0.32 (at 90th pct, 9:1 imbalance) to ~0.87 (at 75th pct, near-balanced)
  - Achieves F1=0.80, Recall=86%, Precision=75% vs. F1=0.33 at the 90th percentile
  - Is still methodologically sound: threshold computed from training set only, on the correct column (`next_month_rat_count`)
  - Does NOT change the inspection list output — we always export the top 200 blocks by risk score regardless of the threshold
- **Do not use KFold CV** — always use `TimeSeriesSplit`; KFold allows future time periods to leak into CV training folds
- **Do not compute threshold on full dataset** — always compute `next_month_rat_count.quantile(RAT_QUANTILE_THRESHOLD)` on the training split only to prevent target leakage
