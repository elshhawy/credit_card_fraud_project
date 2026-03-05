"""
Soft Voting Ensemble builder.

Why soft voting?
    All base models are calibrated → probabilities are reliable
    Soft voting averages probabilities → better than hard voting
    Combines strengths of all 4 models
"""

from sklearn.ensemble import VotingClassifier


def build_voting_model(log_model, rf_model,
                       xgb_model, nn_model,
                       X_train, y_train):
    """
    Build and fit Soft Voting Ensemble.

    Parameters
    ----------
    log_model, rf_model,
    xgb_model, nn_model : fitted calibrated models
    X_train             : np.ndarray
    y_train             : np.ndarray

    Returns
    -------
    model : fitted VotingClassifier
    """
    model = VotingClassifier(
        estimators=[
            ("logistic",       log_model),
            ("random_forest",  rf_model),
            ("xgboost",        xgb_model),
            ("neural_network", nn_model),
        ],
        voting="soft",
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model