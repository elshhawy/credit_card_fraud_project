# ═══════════════════════════════════════════════════════════
# src/feature_engineering.py
# ═══════════════════════════════════════════════════════════
"""
Feature engineering pipeline.

Decisions taken from EDA:
    1. log_amount  = log1p(Amount)
       → Amount has extreme skewness (19.99 → 0.16)

    2. time_sin = sin(2π × Time / time_max)
       time_cos = cos(2π × Time / time_max)
       → Cyclical encoding — no fixed reference assumed
       → time_max must come from Train to avoid leakage

    3. Drop original Amount and Time columns

Functions
---------
apply_feature_engineering() : Apply all transformations
"""

import numpy as np
import pandas as pd


def apply_feature_engineering(X: pd.DataFrame,
                               time_max: float = None) -> pd.DataFrame:
    """
    Apply all feature engineering steps.

    Parameters
    ----------
    X        : pd.DataFrame  features (no target column)
    time_max : float
               Maximum Time value from Train set.
               MUST be passed from Train — never computed
               from Val or Test to avoid data leakage.
               If None, computed from X (use only for Train).

    Returns
    -------
    X : pd.DataFrame  transformed features
    """
    X = X.copy()

    # ── Feature 1: Log transform of Amount ───────────────
    # Amount has extreme skewness (19.99)
    # log1p reduces it to 0.16
    X["log_amount"] = np.log1p(X["Amount"])

    # ── Feature 2: Cyclical encoding of Time ─────────────
    # Time = seconds elapsed since first transaction
    # We don't know the absolute starting point
    # → Cyclical encoding preserves pattern safely
    if time_max is None:
        time_max = X["Time"].max()  # Only safe for Train

    X["time_sin"] = np.sin(2 * np.pi * X["Time"] / time_max)
    X["time_cos"] = np.cos(2 * np.pi * X["Time"] / time_max)

    # ── Drop original columns ─────────────────────────────
    X.drop(columns=["Amount", "Time"], inplace=True)

    return X