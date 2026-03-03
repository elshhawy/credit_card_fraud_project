# ═══════════════════════════════════════════════════════════
# src/config.py
# ═══════════════════════════════════════════════════════════
"""
Central configuration for Credit Fraud Detection project.
All paths, hyperparameters, and settings are defined here.
Import from this file instead of hardcoding values anywhere.
"""

import os
from pathlib import Path

# ══════════════════════════════════════════════════════════
# PATHS
# ══════════════════════════════════════════════════════════

BASE_DIR   = Path(__file__).resolve().parent.parent

DATA_DIR   = BASE_DIR / "data" / "raw"
TRAIN_PATH = DATA_DIR / "train.csv"
VAL_PATH   = DATA_DIR / "val.csv"
TEST_PATH  = DATA_DIR / "test.csv"

OUTPUT_DIR  = BASE_DIR / "outputs"
MODELS_DIR  = OUTPUT_DIR / "models"
FIGURES_DIR = OUTPUT_DIR / "figures"
REPORTS_DIR = OUTPUT_DIR / "reports"

BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"
RESULTS_PATH    = REPORTS_DIR / "model_results.csv"

# Create output directories if they don't exist
for _dir in [MODELS_DIR, FIGURES_DIR, REPORTS_DIR]:
    os.makedirs(_dir, exist_ok=True)

# ══════════════════════════════════════════════════════════
# DATA SETTINGS
# ══════════════════════════════════════════════════════════

TARGET_COLUMN = "Class"
RANDOM_STATE  = 42

# ══════════════════════════════════════════════════════════
# CROSS-VALIDATION
# ══════════════════════════════════════════════════════════

CV_FOLDS   = 5
CV_SCORING = "f1"          # Primary metric = F1

# ══════════════════════════════════════════════════════════
# FEATURE SELECTION
# ══════════════════════════════════════════════════════════

FS_THRESHOLD     = "median"
FS_N_ESTIMATORS  = 100

# ══════════════════════════════════════════════════════════
# MODEL HYPERPARAMETER GRIDS
# ══════════════════════════════════════════════════════════

LOGISTIC_PARAM_GRID = {
    "C": [0.1, 1, 10],
    "penalty": ["l1", "l2"],
}

RF_PARAM_GRID = {
    "n_estimators": [100, 200],
    "max_depth": [5, 10],
    "min_samples_split": [2, 5],
}

XGB_PARAM_GRID = {
    "n_estimators": [100, 200],
    "max_depth": [3, 5],
    "learning_rate": [0.05, 0.1],
}

NN_PARAM_GRID = {
    "hidden_layer_sizes": [(64, 32), (128, 64)],
    "alpha": [0.0001, 0.001],
}

# ══════════════════════════════════════════════════════════
# MODEL BASE PARAMS
# ══════════════════════════════════════════════════════════

LOGISTIC_BASE_PARAMS = dict(
    solver="liblinear",
    class_weight="balanced",
    random_state=RANDOM_STATE,
    max_iter=1000,
)

RF_BASE_PARAMS = dict(
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

XGB_BASE_PARAMS = dict(
    eval_metric="aucpr",
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbosity=0,
)

NN_BASE_PARAMS = dict(
    activation="relu",
    solver="adam",
    max_iter=500,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
    random_state=RANDOM_STATE,
)