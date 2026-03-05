"""
XGBoost model builder.

Steps
-----
1. Compute scale_pos_weight = count(0) / count(1)
   → Handles 559:1 imbalance at model level
   → From EDA: scale_pos_weight = 559.28
2. GridSearchCV
3. Calibration → isotonic regression
"""

from collections import Counter

from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.calibration import CalibratedClassifierCV

from config import (
    XGB_BASE_PARAMS,
    XGB_PARAM_GRID,
    CV_SCORING,
)


def get_scale_pos_weight(y) -> float:
    """
    Compute scale_pos_weight for XGBoost.
    Formula: count(negatives) / count(positives)
    """
    c = Counter(y)
    return c[0] / c[1]


def build_xgb_model(X_train, y_train, cv_strategy):
    """
    Train XGBoost with scale_pos_weight + GridSearchCV + Calibration.

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
    spw    = get_scale_pos_weight(y_train)
    params = {**XGB_BASE_PARAMS, "scale_pos_weight": spw}

    base   = XGBClassifier(**params)

    grid   = GridSearchCV(
        base,
        XGB_PARAM_GRID,
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