"""
Preprocessing algorithms
======================================

All algotirhms should take inputs as a 3D tensor, and returns processed
3D tensor.
"""

from __future__ import division, print_function, absolute_import

from nnmnkwii.utils import dimention_wise_delta, trim_zeros_frames
import numpy as np


class UtteranceWiseTransformer(object):
    def transform(self, X):
        assert X.ndim == 3
        N, T, D = X.shape
        Y = np.zeros(self.get_shape(X), dtype=X.dtype)
        for idx, x in enumerate(X):
            x = trim_zeros_frames(x)
            y = self.do_transform(x)
            Y[idx][:len(y)] = y
        return Y

    def get_shape(self, X):
        raise NotImplementedError


class SilenceTrim(UtteranceWiseTransformer):
    def __init__(self, power_func, threshold=-10):
        self.power_func = power_func
        self.threshold = threshold

    def get_shape(self, X):
        return X.shape

    def do_transform(self, x):
        T, D = x.shape
        y = x.copy()

        def __trim_inplace(x, indices, power_func, th):
            for t in indices:
                power = power_func(x[t])
                if power < th:
                    x[t, :] = 0
                else:
                    break

        # Forward
        __trim_inplace(y, range(len(x)), self.power_func, self.threshold)
        # Backward
        __trim_inplace(y, range(len(x) - 1, 0, -1), self.power_func,
                       self.threshold)
        return y


class DeltaAppender(UtteranceWiseTransformer):
    def __init__(self, windows):
        self.windows = windows

    def get_shape(self, X):
        N, T, D = X.shape
        return (N, T, D * len(self.windows))

    def do_transform(self, x):
        features = []
        for _, _, window in self.windows:
            features.append(dimention_wise_delta(x, window))
        combined_features = np.hstack(features)
        return combined_features
