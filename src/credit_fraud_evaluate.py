# ═══════════════════════════════════════════════════════════
# src/credit_fraud_evaluate.py
# ═══════════════════════════════════════════════════════════
"""
Final Test Evaluation for Credit Fraud Detection.

Rules
-----
- Test set is used ONCE only — here
- Threshold comes from saved model (derived from Validation)
- No decisions are made based on Test results
- This script runs AFTER credit_fraud_train.py

Usage
-----
    cd CREDIT_FRAUD_PROJECT
    python src/credit_fraud_evaluate.py
"""

import sys
import pickle
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import BEST_MODEL_PATH
from credit_fraud_utils_data import load_data, split_features_target
from feature_engineering import apply_feature_engineering
from credit_fraud_utils_eval import (
    evaluate_model,
    plot_pr_curve,
    plot_confusion_matrix,
)


# ══════════════════════════════════════════════════════════
# STEP 1 — Load Best Model
# ══════════════════════════════════════════════════════════
print("\n" + "="*55)
print("  CREDIT FRAUD DETECTION — FINAL TEST EVALUATION")
print("="*55)

print(f"\n  [STEP 1] Loading best model...")
print(f"  Path: {BEST_MODEL_PATH}")

with open(BEST_MODEL_PATH, "rb") as f:
    payload = pickle.load(f)

model         = payload["model"]
selector      = payload["selector"]
scaler        = payload["scaler"]
time_max      = payload["time_max"]
threshold     = payload["threshold"]
model_name    = payload["model_name"]
sampling      = payload["sampling"]
val_f1        = payload["f1"]
val_pr_auc    = payload["pr_auc"]
val_roc_auc   = payload["roc_auc"]

print(f"\n  Model    : {model_name}")
print(f"  Sampling : {sampling}")
print(f"\n  Validation Performance (from training):")
print(f"  F1-Score : {val_f1:.4f}  ← Primary Metric")
print(f"  PR-AUC   : {val_pr_auc:.4f}")
print(f"  ROC-AUC  : {val_roc_auc:.4f}")
print(f"  Threshold: {threshold:.4f}")


# ══════════════════════════════════════════════════════════
# STEP 2 — Load and Prepare Test Data
# ══════════════════════════════════════════════════════════
print(f"\n  [STEP 2] Loading and preparing Test data...")

_, _, test_df    = load_data()
X_test_raw, y_test = split_features_target(test_df)

# Feature engineering — same time_max from Train
X_test_eng    = apply_feature_engineering(
    X_test_raw, time_max=time_max
)

# Scaling — transform only, scaler was fitted on Train
X_test_scaled = scaler.transform(X_test_eng)

# Feature selection — same selector from training
X_test_sel    = selector.transform(X_test_scaled)

print(f"  Test shape after preparation: {X_test_sel.shape}")


# ══════════════════════════════════════════════════════════
# STEP 3 — Final Test Evaluation
# ══════════════════════════════════════════════════════════
print(f"\n  [STEP 3] Final Test Evaluation...")
print(f"  ⚠️  Test set is being used for the first time")

test_probs = model.predict_proba(X_test_sel)[:, 1]

test_f1, test_pr_auc, test_roc_auc = evaluate_model(
    y_test,
    test_probs,
    threshold,
    label=f"FINAL TEST — {model_name} | {sampling}",
    verbose=True,
)


# ══════════════════════════════════════════════════════════
# STEP 4 — Save Figures
# ══════════════════════════════════════════════════════════
print(f"\n  [STEP 4] Saving figures...")

plot_pr_curve(
    y_test, test_probs,
    label=f"{model_name} | {sampling}",
    save_name="pr_curve_test.png",
)

plot_confusion_matrix(
    y_test, test_probs, threshold,
    label=f"{model_name} | {sampling}",
    save_name="confusion_matrix_test.png",
)


# ══════════════════════════════════════════════════════════
# STEP 5 — Validation vs Test Comparison
# ══════════════════════════════════════════════════════════
print(f"\n{'='*55}")
print(f"  VALIDATION vs TEST COMPARISON")
print(f"  Model    : {model_name}")
print(f"  Sampling : {sampling}")
print(f"{'='*55}")
print(f"  Metric    | Validation | Test")
print(f"  {'─'*35}")
print(f"  F1-Score  | {val_f1:.4f}     | {test_f1:.4f}")
print(f"  PR-AUC    | {val_pr_auc:.4f}     | {test_pr_auc:.4f}")
print(f"  ROC-AUC   | {val_roc_auc:.4f}     | {test_roc_auc:.4f}")
print(f"  Threshold | {threshold:.4f}     | {threshold:.4f}")
print(f"{'='*55}")

# Check for overfitting
f1_diff = abs(val_f1 - test_f1)
if f1_diff < 0.02:
    print(f"\n  ✅ No overfitting — F1 difference: {f1_diff:.4f}")
elif f1_diff < 0.05:
    print(f"\n  ⚠️  Slight difference — F1 diff: {f1_diff:.4f}")
else:
    print(f"\n  ❌ Large difference — possible overfitting: {f1_diff:.4f}")

print(f"\n  ✅ Evaluation complete!")