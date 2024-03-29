import numpy as np

# Compat
from nnmnkwii.preprocessing import adjust_frame_length  # noqa
from nnmnkwii.preprocessing import delta_features  # noqa
from nnmnkwii.preprocessing import meanstd  # noqa
from nnmnkwii.preprocessing import meanvar  # noqa
from nnmnkwii.preprocessing import minmax  # noqa
from nnmnkwii.preprocessing import minmax_scale  # noqa
from nnmnkwii.preprocessing import remove_zeros_frames  # noqa
from nnmnkwii.preprocessing import scale  # noqa
from nnmnkwii.preprocessing import trim_zeros_frames  # noqa

apply_delta_windows = delta_features

from .files import *  # noqa


def apply_each2d_trim(func2d, X, *args, **kwargs):
    """Apply function for each trimmed 2d slice.

    Args:
        func2d (Function): Function applied multiple times for each 2d slice.
        X (numpy.ndarray): Input 3d array of shape (``N x T x D``)

    Returns:
        numpy.ndarray: Output array (``N x T x D'``)
    """
    assert X.ndim == 3
    N, T, _ = X.shape
    x = trim_zeros_frames(X[0])
    y = func2d(x, *args, **kwargs)
    assert y.ndim == 2
    _, D = y.shape

    Y = np.zeros((N, T, D))
    for idx in range(N):
        x = trim_zeros_frames(X[idx])
        y = func2d(x, *args, **kwargs)
        Y[idx][: len(y)] = y
    return Y


def apply_each2d_padded(func2d, X, lengths, *args, **kwargs):
    """Apply function for each padded 2d slice.

    Args:
        func2d (Function): Function applied multiple times for each 2d slice.
        X (numpy.ndarray): Input 3d array of shape (``N x T x D``)
        lengths (array_like): Lengths for each 2d slice

    Returns:
        numpy.ndarray: Output array (``N x T x D'``)
    """
    assert X.ndim == 3
    N, T, _ = X.shape
    y = func2d(X[0][: lengths[0]], *args, **kwargs)
    assert y.ndim == 2
    _, D = y.shape

    Y = np.zeros((N, T, D))
    Y[0][: len(y)] = y
    for idx in range(1, N):
        y = func2d(X[idx][: lengths[idx]], *args, **kwargs)
        Y[idx][: len(y)] = y
    return Y
