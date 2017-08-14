# coding: utf-8
from __future__ import division, print_function, absolute_import

from nnmnkwii.util import delta, trim_zeros_frames
import numpy as np


# TODO: Is this really needed? Isn't decorator sufficient?
class UtteranceWiseTransformer(object):
    def transform(self, X, lengths=None):
        assert X.ndim == 3
        N, T, D = X.shape
        Y = np.zeros(self.get_shape(X), dtype=X.dtype)
        for idx, x in enumerate(X):
            if lengths is None:
                x = trim_zeros_frames(x)
            else:
                x = x[:lengths[idx]]
            y = self.do_transform(x)
            Y[idx][:len(y)] = y
        return Y

    def get_shape(self, X):
        raise NotImplementedError


class DeltaAppender(UtteranceWiseTransformer):
    """Append delta features.

    Given a ``N x T x D`` array, features of multiple utterances,
    transform features into static + delta features for each utterance.

    Attributes:
        windows (list): A sequence of windows. See
          :func:`nnmnkwii.functions.mlpg` for what window means.
    """

    def __init__(self, windows):
        self.windows = windows

    def get_shape(self, X):
        N, T, D = X.shape
        return (N, T, D * len(self.windows))

    def do_transform(self, x):
        features = []
        for _, _, window in self.windows:
            features.append(delta(x, window))
        combined_features = np.hstack(features)
        return combined_features
