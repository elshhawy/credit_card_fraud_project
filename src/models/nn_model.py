# ═══════════════════════════════════════════════════════════
# src/models/nn_model.py
# ═══════════════════════════════════════════════════════════
"""
Neural Network (MLP) model builder.

Steps
-----
1. GridSearchCV  → find best architecture and alpha
2. Calibration   → Platt scaling (sigmoid)
                   standard calibration for NNs
"""

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.calibration import CalibratedClassifierCV

from config import (
    NN_BASE_PARAMS,
    NN_PARAM_GRID,
    CV_SCORING,
)


def build_nn_model(X_train, y_train, cv_strategy):
    """
    Train Neural Network with GridSearchCV + Calibration.

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
    base = MLPClassifier(**NN_BASE_PARAMS)

    grid = GridSearchCV(
        base,
        NN_PARAM_GRID,
        scoring=CV_SCORING,
        cv=cv_strategy,
        n_jobs=-1,
        verbose=0,
    )
    grid.fit(X_train, y_train)

    # Platt scaling → standard calibration for NNs
    model = CalibratedClassifierCV(
        grid.best_estimator_, cv=3, method="sigmoid"
    )
    model.fit(X_train, y_train)

    return model, grid.best_params_