# ═══════════════════════════════════════════════════════════
# src/models/rf_model.py
# ═══════════════════════════════════════════════════════════
"""
Random Forest model builder.

Steps
-----
1. GridSearchCV  → find best n_estimators, max_depth, etc.
2. Calibration   → isotonic regression
                   (RF probs are often overconfident)
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.calibration import CalibratedClassifierCV

from config import (
    RF_BASE_PARAMS,
    RF_PARAM_GRID,
    CV_SCORING,
)


def build_rf_model(X_train, y_train, cv_strategy):
    """
    Train Random Forest with GridSearchCV + Calibration.

    Parameters
    ----------
    X_train     : np.ndarray
    y_train     : np.ndarray
    cv_strategy : StratifiedKFold

    Returns
    -------
    model       : fitted calibrated model
    best_params : dict
    """
    base = RandomForestClassifier(**RF_BASE_PARAMS)

    grid = GridSearchCV(
        base,
        RF_PARAM_GRID,
        scoring=CV_SCORING,
        cv=cv_strategy,
        n_jobs=-1,
        verbose=0,
    )
    grid.fit(X_train, y_train)

    model = CalibratedClassifierCV(
        grid.best_estimator_, cv=3, method="isotonic"
    )
    model.fit(X_train, y_train)

    return model, grid.best_params_