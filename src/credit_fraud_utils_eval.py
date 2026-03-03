# ═══════════════════════════════════════════════════════════
# src/credit_fraud_utils_eval.py
# ═══════════════════════════════════════════════════════════
"""
Evaluation utilities for Credit Fraud Detection.

Functions
---------
find_best_threshold() : Best threshold via PR Curve (max F1)
evaluate_model()      : Compute F1, PR-AUC, ROC-AUC + report
plot_pr_curve()       : Plot and save PR Curve
save_results()        : Save results DataFrame to CSV
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    f1_score,
    average_precision_score,
    precision_recall_curve,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    ConfusionMatrixDisplay,
)

from config import FIGURES_DIR, RESULTS_PATH


# ══════════════════════════════════════════════════════════
# THRESHOLD SELECTION
# ══════════════════════════════════════════════════════════

def find_best_threshold(y_true, probs):
    """
    Find threshold that maximizes F1-score.

    Why PR Curve?
    → Computes F1 at every possible threshold at once
    → More reliable than looping over a fixed grid
    → Never touches Test set

    Parameters
    ----------
    y_true : array-like  ground truth labels
    probs  : array-like  predicted probabilities for class 1

    Returns
    -------
    best_threshold : float
    best_f1        : float
    """
    precisions, recalls, thresholds = precision_recall_curve(
        y_true, probs
    )

    # F1 = 2PR / (P+R) at every threshold
    f1_scores = np.where(
        (precisions + recalls) == 0,
        0,
        2 * precisions * recalls / (precisions + recalls)
    )

    # thresholds is 1 element shorter than precisions/recalls
    best_idx       = np.argmax(f1_scores[:-1])
    best_threshold = float(thresholds[best_idx])
    best_f1        = float(f1_scores[best_idx])

    return best_threshold, best_f1


# ══════════════════════════════════════════════════════════
# EVALUATION
# ══════════════════════════════════════════════════════════

def evaluate_model(y_true, probs, threshold, label="", verbose=True):
    """
    Compute all metrics and print full report.

    Primary metric = F1 Score

    Parameters
    ----------
    y_true    : array-like
    probs     : array-like
    threshold : float
    label     : str   printed in report header
    verbose   : bool  whether to print report

    Returns
    -------
    f1, pr_auc, roc_auc : float
    """
    preds   = (probs >= threshold).astype(int)
    f1      = f1_score(y_true, preds, zero_division=0)
    pr_auc  = average_precision_score(y_true, probs)
    roc_auc = roc_auc_score(y_true, probs)

    if verbose:
        cm = confusion_matrix(y_true, preds)
        print(f"\n{'─'*50}")
        if label:
            print(f"  {label}")
        print(f"  Threshold  : {threshold:.4f}")
        print(f"  F1-Score   : {f1:.4f}   ← Primary Metric")
        print(f"  PR-AUC     : {pr_auc:.4f}")
        print(f"  ROC-AUC    : {roc_auc:.4f}")
        print(f"\n  Confusion Matrix:")
        print(f"  TN={cm[0,0]:,}  FP={cm[0,1]:,}")
        print(f"  FN={cm[1,0]:,}  TP={cm[1,1]:,}")
        print(f"\n{classification_report(y_true, preds, zero_division=0)}")

    return f1, pr_auc, roc_auc


# ══════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════

def plot_pr_curve(y_true, probs, label="", save_name="pr_curve.png"):
    """
    Plot and save Precision-Recall Curve.

    Parameters
    ----------
    y_true    : array-like
    probs     : array-like
    label     : str  shown in legend
    save_name : str  filename in outputs/figures/
    """
    precisions, recalls, _ = precision_recall_curve(y_true, probs)
    ap = average_precision_score(y_true, probs)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recalls, precisions, lw=2, color="crimson",
            label=f"{label}  (AP = {ap:.4f})")
    ax.set_title("Precision-Recall Curve", fontweight="bold")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    save_path = FIGURES_DIR / save_name
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ PR curve saved → {save_path}")


def plot_confusion_matrix(y_true, probs, threshold,
                           label="",
                           save_name="confusion_matrix.png"):
    """
    Plot and save Confusion Matrix.

    Parameters
    ----------
    y_true    : array-like
    probs     : array-like
    threshold : float
    label     : str
    save_name : str
    """
    preds = (probs >= threshold).astype(int)
    cm    = confusion_matrix(y_true, preds)

    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(
        cm, display_labels=["Legitimate", "Fraud"]
    )
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix\n{label}", fontweight="bold")
    plt.tight_layout()

    save_path = FIGURES_DIR / save_name
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ Confusion matrix saved → {save_path}")


# ══════════════════════════════════════════════════════════
# SAVE RESULTS
# ══════════════════════════════════════════════════════════

def save_results(results: list) -> pd.DataFrame:
    """
    Convert results list to DataFrame, sort by F1, save to CSV.

    Parameters
    ----------
    results : list of dicts

    Returns
    -------
    results_df : pd.DataFrame  sorted by f1 descending
    """
    df = pd.DataFrame(results)
    df = df.sort_values("f1", ascending=False).reset_index(drop=True)
    df.to_csv(RESULTS_PATH, index=False)
    print(f"  ✅ Results saved → {RESULTS_PATH}")
    return df