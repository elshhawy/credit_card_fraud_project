"""
Sampling strategies for imbalanced classification.

Why sampling?
    Train set has 559:1 imbalance (legitimate:fraud)
    Sampling helps models learn fraud patterns better

Strategies
----------
none          : No resampling — baseline
random_over   : Randomly duplicate minority samples
random_under  : Randomly remove majority samples
smote         : Synthetic minority oversampling
adasyn        : Adaptive synthetic sampling
smote_tomek   : SMOTE + clean boundary (Tomek links)
smote_enn     : SMOTE + aggressive cleaning (ENN)

Functions
---------
get_samplers() : Returns dict of {name: sampler}
"""

from imblearn.over_sampling import (
    SMOTE,
    ADASYN,
    RandomOverSampler,
)
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek, SMOTEENN

from config import RANDOM_STATE


def get_samplers() -> dict:
    """
    Return all sampling strategies.

    Returns
    -------
    samplers : dict  {name: sampler or None}
               None means no resampling (baseline)
    """
    return {
        "none": None,

        "random_over": RandomOverSampler(
            random_state=RANDOM_STATE
        ),

        "random_under": RandomUnderSampler(
            random_state=RANDOM_STATE
        ),

        "smote": SMOTE(
            k_neighbors=5,
            random_state=RANDOM_STATE,
        ),

        "adasyn": ADASYN(
            n_neighbors=5,
            random_state=RANDOM_STATE,
        ),

        "smote_tomek": SMOTETomek(
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),

        "smote_enn": SMOTEENN(
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }