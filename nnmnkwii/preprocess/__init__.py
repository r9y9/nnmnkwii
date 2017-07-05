from __future__ import division, print_function, absolute_import

"""Commonly used preprocessing algorithms

All algotirhms should take inputs as a 3D tensor, and returns processed 3D tensor.
"""

from nnmnkwii.utils import dimention_wise_delta, trim_zeros_frames
import numpy as np


class SilenceRemoval(object):
    pass


class DeltaAppender(object):
    def __init__(self, order=1, win=None):
        self.order = order
        if win is None:
            win = np.array([0.5, 0, -0.5], dtype=np.float32)
        self.win = win

    def transform(self, X):
        assert X.ndim == 3
        N, T, D = X.shape
        Y = np.zeros((N, T, D * (self.order + 1)), dtype=X.dtype)
        for idx, x in enumerate(X):
            features = [trim_zeros_frames(x)]
            for _ in range(self.order):
                dynamic_features = dimention_wise_delta(features[-1], self.win)
                features.append(dynamic_features)
            combined_features = np.hstack(features)
            assert combined_features.shape[-1] == Y.shape[-1]
            Y[idx][:len(combined_features)] = combined_features

        return Y
