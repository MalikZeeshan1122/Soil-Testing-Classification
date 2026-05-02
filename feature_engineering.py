"""Shared feature-engineering helpers used by train, predict, and the app.

Defined in a stable module path so a saved pipeline that pickles a
FunctionTransformer pointing at `add_engineered_features` can be loaded back
from any entry point.
"""

import numpy as np
import pandas as pd


PH_BIN_EDGES = [-np.inf, 6.0, 6.5, 7.0, 7.5, np.inf]
PH_BIN_LABELS = ["vlow", "low", "mid", "high", "vhigh"]


def add_engineered_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    X["ph_sq"] = X["ph"] ** 2
    X["ph_bin"] = pd.cut(X["ph"], bins=PH_BIN_EDGES, labels=PH_BIN_LABELS).astype(str)
    return X
