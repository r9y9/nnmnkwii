from __future__ import division, print_function, absolute_import

"""Commonly used preprocessing algorithms

All algotirhms should take inputs as a 3D tensor, and returns processed 3D tensor.
"""

from nnmnkwii.utils import dimention_wise_delta, trim_zeros_frames
import numpy as np


class SilenceRemoval(object):
    pass


class DeltaAppender(object):
    def __init__(self, windows):
        self.windows = windows

    def transform(self, X):
        assert X.ndim == 3
        N, T, D = X.shape
        Y = np.zeros((N, T, D * len(self.windows)), dtype=X.dtype)
        for idx, x in enumerate(X):
            x = trim_zeros_frames(x)
            features = []
            for _, _, window in self.windows:
                features.append(dimention_wise_delta(x, window))
            combined_features = np.hstack(features)
            assert combined_features.shape[-1] == Y.shape[-1]
            Y[idx][:len(combined_features)] = combined_features

        return Y
