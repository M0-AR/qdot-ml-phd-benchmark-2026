"""Group-safe benchmark protocol (no device leakage)."""
from __future__ import annotations
import numpy as np
from sklearn.model_selection import GroupKFold


def group_cv_indices(devices, n_splits=4, seed=0):
    gkf = GroupKFold(n_splits=n_splits)
    X_dummy = np.zeros(len(devices))
    return list(gkf.split(X_dummy, groups=devices))
