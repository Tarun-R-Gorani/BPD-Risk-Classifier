"""Small, importable helpers used by persisted BPD model pipelines."""

import numpy as np


def variance_scores(X, y):
    """Return feature variances for fold-local unsupervised gene filtering."""
    return np.nanvar(X, axis=0), None
