# src/config.py
# Centralized configuration for the NYC Rat Risk ML pipeline.

# --- API Configuration ---
API_LIMIT = 500_000
RELIABLE_START_DATE = '2022-01-01'
COORDINATE_PRECISION = 4
SOCRATA_DATE_FORMAT = '%Y-%m-%dT%H:%M:%S.%f'

# --- Model Configuration ---
RAT_QUANTILE_THRESHOLD = 0.75
TRAIN_SPLIT_RATIO = 0.8
CV_FOLDS = 3
TOP_N_MARKERS = 200

# --- Feature Lists ---
COUNT_FEATURES = [
    'lag_1_rat', 'lag_1_dumping', 'lag_2_rat', 'lag_6_rat',
    'lag_3_avg_rat', 'lag_12_avg_rat', 'dumping_rat_interaction'
]
TIME_FEATURES = ['month', 'year']
BOROUGH_COL = 'Borough'
TARGET = 'TARGET_HIGH_RISK'
