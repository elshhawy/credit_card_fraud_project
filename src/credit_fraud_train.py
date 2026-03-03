# ═══════════════════════════════════════════════════════════
# src/credit_fraud_train.py
# ═══════════════════════════════════════════════════════════
"""
Main training pipeline for Credit Fraud Detection.

Best combination was determined from notebook experiments:
    Model    : XGBoost
    Sampling : None (scale_pos_weight=559.28 was sufficient)
    F1       : 0.8757 on Validation

What this script does:
    1. Load Train and Val data
    2. Feature engineering (log_amount, time_sin, time_cos)
    3. StandardScaler — fit on Train only
    4. Feature Selection — fit on Train only
    5. Train XGBoost with best hyperparameters
    6. Evaluate on Validation
    7. Save best model PKL with all details

Usage
-----
    cd CREDIT_FRAUD_PROJECT
    python src/credit_fraud_train.py
"""

import sys
import pickle
import warnings
from pathlib import Path
from collections import Counter

import numpy as np

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score,
    average_precision_score,
    roc_auc_score,
)

from config import (
    RANDOM_STATE,
    CV_FOLDS,
    FS_THRESHOLD,
    FS_N_ESTIMATORS,
    BEST_MODEL_PATH,
)
from credit_fraud_utils_data import (
    load_data,
    split_features_target,
    scale_data,
)
from feature_engineering import apply_feature_engineering
from credit_fraud_utils_eval import find_best_threshold
from models import build_xgb_model


# ══════════════════════════════════════════════════════════
# BEST COMBINATION — determined from notebook experiments
# ══════════════════════════════════════════════════════════
# To change the best model, update these two lines only
BEST_MODEL_TYPE = "xgboost"
BEST_SAMPLER    = "none"
# scale_pos_weight handles imbalance — no sampling needed


# ══════════════════════════════════════════════════════════
# STEP 1 — Load Data
# ══════════════════════════════════════════════════════════
print("\n" + "="*55)
print("  CREDIT FRAUD DETECTION — TRAINING PIPELINE")
print("="*55)

print("\n  [STEP 1] Loading Data...")
train_df, val_df, _ = load_data()

X_train_raw, y_train = split_features_target(train_df)
X_val_raw,   y_val   = split_features_target(val_df)


# ══════════════════════════════════════════════════════════
# STEP 2 — Feature Engineering
# ══════════════════════════════════════════════════════════
print("\n  [STEP 2] Feature Engineering...")

# time_max from Train ONLY — avoid leakage
time_max = X_train_raw["Time"].max()

X_train_eng = apply_feature_engineering(X_train_raw, time_max=time_max)
X_val_eng   = apply_feature_engineering(X_val_raw,   time_max=time_max)

feature_names = list(X_train_eng.columns)
print(f"  Features after engineering : {len(feature_names)}")


# ══════════════════════════════════════════════════════════
# STEP 3 — Scaling
# ══════════════════════════════════════════════════════════
print("\n  [STEP 3] Scaling...")

X_train_scaled, X_val_scaled, _, scaler = scale_data(
    X_train_eng, X_val_eng, X_val_eng
)


# ══════════════════════════════════════════════════════════
# STEP 4 — Feature Selection
# ══════════════════════════════════════════════════════════
print("\n  [STEP 4] Feature Selection...")

fs_rf = RandomForestClassifier(
    n_estimators=FS_N_ESTIMATORS,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
fs_rf.fit(X_train_scaled, y_train)

selector  = SelectFromModel(fs_rf, threshold=FS_THRESHOLD, prefit=True)
X_train_sel = selector.transform(X_train_scaled)
X_val_sel   = selector.transform(X_val_scaled)

selected_features = [
    feature_names[i]
    for i in selector.get_support(indices=True)
]
print(f"  Features: {X_train_scaled.shape[1]} → {X_train_sel.shape[1]} selected")
print(f"  Selected : {selected_features}")


# ══════════════════════════════════════════════════════════
# STEP 5 — Train Best Model
# ══════════════════════════════════════════════════════════
print(f"\n  [STEP 5] Training {BEST_MODEL_TYPE.upper()} "
      f"with sampling={BEST_SAMPLER}...")

cv_strategy = StratifiedKFold(
    n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE
)

# No sampling — scale_pos_weight handles imbalance
print(f"  Class distribution: {Counter(y_train)}")

model, best_params = build_xgb_model(
    X_train_sel, y_train, cv_strategy
)
print(f"  Best params : {best_params}")


# ══════════════════════════════════════════════════════════
# STEP 6 — Evaluate on Validation
# ══════════════════════════════════════════════════════════
print("\n  [STEP 6] Evaluating on Validation...")

val_probs = model.predict_proba(X_val_sel)[:, 1]
threshold, _ = find_best_threshold(y_val, val_probs)
val_preds    = (val_probs >= threshold).astype(int)

val_f1      = f1_score(y_val, val_preds, zero_division=0)
val_pr_auc  = average_precision_score(y_val, val_probs)
val_roc_auc = roc_auc_score(y_val, val_probs)

print(f"\n{'='*55}")
print(f"  VALIDATION RESULTS")
print(f"{'='*55}")
print(f"  Model     : {BEST_MODEL_TYPE}")
print(f"  Sampling  : {BEST_SAMPLER}")
print(f"  F1-Score  : {val_f1:.4f}  ← Primary Metric")
print(f"  PR-AUC    : {val_pr_auc:.4f}")
print(f"  ROC-AUC   : {val_roc_auc:.4f}")
print(f"  Threshold : {threshold:.4f}")
print(f"  Features  : {X_train_sel.shape[1]}")
print(f"{'='*55}")


# ══════════════════════════════════════════════════════════
# STEP 7 — Save Best Model
# ══════════════════════════════════════════════════════════
print("\n  [STEP 7] Saving Best Model...")

payload = {
    "model":             model,
    "threshold":         threshold,
    "selector":          selector,
    "scaler":            scaler,
    "time_max":          time_max,
    "feature_names":     feature_names,
    "model_name":        BEST_MODEL_TYPE,
    "sampling":          BEST_SAMPLER,
    "best_params":       best_params,
    "f1":                val_f1,
    "pr_auc":            val_pr_auc,
    "roc_auc":           val_roc_auc,
    "selected_features": selected_features,
    "n_features":        X_train_sel.shape[1],
}

with open(BEST_MODEL_PATH, "wb") as f:
    pickle.dump(payload, f)

print(f"  ✅ Best model saved → {BEST_MODEL_PATH}")
print(f"  ✅ Training complete!")