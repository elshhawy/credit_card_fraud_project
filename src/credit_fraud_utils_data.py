"""
Data loading and preprocessing utilities.

Functions
---------
load_data()             : Load train, val, test CSV files
split_features_target() : Separate X and y from dataframe
scale_data()            : Fit scaler on Train, transform all
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from config import (
    TRAIN_PATH,
    VAL_PATH,
    TEST_PATH,
    TARGET_COLUMN,
)


# ══════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════

def load_data():
    """
    Load train, val, test from paths defined in config.

    Returns
    -------
    train_df, val_df, test_df : pd.DataFrame
    """
    train_df = pd.read_csv(TRAIN_PATH)
    val_df   = pd.read_csv(VAL_PATH)
    test_df  = pd.read_csv(TEST_PATH)

    # Print Train info only — no leakage from Val/Test
    print(f"  Train : {train_df.shape} | "
          f"Fraud: {train_df[TARGET_COLUMN].sum()} "
          f"({train_df[TARGET_COLUMN].mean()*100:.3f}%)")
    print(f"  Val   : {val_df.shape}   ✅ loaded")
    print(f"  Test  : {test_df.shape}  ✅ loaded")

    # Sanity check — columns must match
    assert list(train_df.columns) == \
           list(val_df.columns)   == \
           list(test_df.columns), \
           "❌ Column mismatch across sets!"

    print("  ✅ All sets have identical columns")

    return train_df, val_df, test_df


# ══════════════════════════════════════════════════════════
# SPLIT FEATURES AND TARGET
# ══════════════════════════════════════════════════════════

def split_features_target(df: pd.DataFrame):
    """
    Split dataframe into X (features) and y (target).

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    X : pd.DataFrame
    y : np.ndarray
    """
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].values
    return X, y


# ══════════════════════════════════════════════════════════
# SCALING
# ══════════════════════════════════════════════════════════

def scale_data(X_train: pd.DataFrame,
               X_val:   pd.DataFrame,
               X_test:  pd.DataFrame):
    """
    Fit StandardScaler on Train ONLY.
    Apply same scaler to Val and Test.

    Parameters
    ----------
    X_train, X_val, X_test : pd.DataFrame (after feature engineering)

    Returns
    -------
    X_train_scaled : np.ndarray
    X_val_scaled   : np.ndarray
    X_test_scaled  : np.ndarray
    scaler         : fitted StandardScaler
    """
    scaler = StandardScaler()

    # fit on Train only
    X_train_scaled = scaler.fit_transform(X_train)

    # Val/Test receive transform only — scaler parameters come from Train
    X_val_scaled   = scaler.transform(X_val)
    X_test_scaled  = scaler.transform(X_test)

    print(f"  ✅ Scaler fitted on Train only")
    print(f"  Train scaled : {X_train_scaled.shape}")

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler