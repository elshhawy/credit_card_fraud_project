# ═══════════════════════════════════════════════════════════
# src/models/logistic_model.py
# ═══════════════════════════════════════════════════════════
"""
Logistic Regression model builder.

Steps
-----
1. GridSearchCV  → find best C and penalty
2. Calibration   → isotonic regression for reliable probs
"""

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.calibration import CalibratedClassifierCV

from config import (
    LOGISTIC_BASE_PARAMS,
    LOGISTIC_PARAM_GRID,
    CV_SCORING,
)


def build_logistic_model(X_train, y_train, cv_strategy):
    """
    Train Logistic Regression with GridSearchCV + Calibration.

    Parameters
    ----------
    X_train     : np.ndarray  (after sampling + feature selection)
    y_train     : np.ndarray
    cv_strategy : StratifiedKFold

    Returns
    -------
    model       : fitted calibrated model
    best_params : dict
    """
    base = LogisticRegression(**LOGISTIC_BASE_PARAMS)

    grid = GridSearchCV(
        base,
        LOGISTIC_PARAM_GRID,
        scoring=CV_SCORING,
        cv=cv_strategy,
        n_jobs=-1,
        verbose=0,
    )
    grid.fit(X_train, y_train)

    # Isotonic calibration → reliable probabilities
    model = CalibratedClassifierCV(
        grid.best_estimator_, cv=3, method="isotonic"
    )
    model.fit(X_train, y_train)

    return model, grid.best_params_