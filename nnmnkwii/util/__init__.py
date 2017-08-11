from __future__ import division, print_function, absolute_import

import numpy as np
from sklearn.utils.extmath import _incremental_mean_and_var
from sklearn.preprocessing.data import _handle_zeros_in_scale

from .files import *


def _delta(x, window):
    return np.correlate(x, window, mode="same")


def delta(x, window):
    """Returns delta features given a static features and a window.

    Args:
        x (ndarray): Input static features (``T x D``)
        window (tuple): A window. See :func:`nnmnkwii.functions.mlpg`.

    Returns:
        (ndarray): Delta features (``T x D``).
    """
    T, D = x.shape
    y = np.zeros_like(x)
    for d in range(D):
        y[:, d] = _delta(x[:, d], window)
    return y


def apply_delta_windows(x, windows):
    """Apply delta windows and combine them.

    This function computes delta features given delta windows, and then
    returns combined features (e.g., static + delta + deltadelta).
    Note that if you want to keep static features, you need to supply
    static window as well as delta windows.

    Args:
        x (2darray): Input static features (shape:  ``T x D``).
        y (list): List of windows. See :func:`nnmnkwii.functions.mlpg` for what
            the delta window means.

    Returns:
        2darray: static + delta features (``T x (D * len(windows)``).

    Examples:
        >>> from nnmnkwii.util import apply_delta_windows
        >>> windows = [
        ...         (0, 0, np.array([1.0])),            # static
        ...         (1, 1, np.array([-0.5, 0.0, 0.5])), # delta
        ...         (1, 1, np.array([1.0, -2.0, 1.0])), # delta-delta
        ...     ]
        >>> T, static_dim = 10, 24
        >>> x = np.random.rand(T, static_dim)
        >>> y = apply_delta_windows(x, windows)
        >>> assert y.shape == (T, static_dim * len(windows))
    """
    T, D = x.shape
    assert len(windows) > 0
    combined_features = np.empty((T, D * len(windows)), dtype=x.dtype)
    for idx, (_, _, window) in enumerate(windows):
        combined_features[:, D * idx:D * idx +D] = delta(x, window)
    return combined_features


def trim_zeros_frames(x, eps=1e-7):
    """Remove trailling zeros frames

    Similar to :func:`numpy.trim_zeros`, trimming trailing zeros features.

    Args:
        x (ndarray): Feature matrix, shape (``T`` x ``D``)

    Returns:
        ndarray: Trimmed 2d feature matrix, shape (``T'`` x ``D``)
    """

    T, D = x.shape
    s = np.sum(np.abs(x), axis=1)
    s[s < eps] = 0.
    return x[:len(np.trim_zeros(s))]


def remove_zeros_frames(x, eps=1e-7):
    """Remove zeros frames

    Given a feature matrix, remove all zeros frames as well as trailing ones.

    Args:
        x (2darray): 2d feature matrix shape (``T`` ,``D``)

    Returns:
        2darray: Zeros-removed 2d feature matrix
    """
    T, D = x.shape
    s = np.sum(np.abs(x), axis=1)
    s[s < eps] = 0.
    return x[s > eps]


def adjast_frame_length(x, y, pad=True, ensure_even=False):
    """Adjast frame lenght given two feature matrices.

    This ensures that two feature matrices have same number of frames, by
    padding zeros to the end or removing last frames.

    Args:
        x (ndarray): Input 2d feature matrix
        y (ndarray): Input 2d feature matrix
        pad (bool) : If True, pads zeros to the end, otherwise removes last few
            frames to ensure same frame lengths.
        ensure_even (bool) : If True, ensure number of frames to be even number.

    Returns:
        Tuple: Pair of adjasted feature matrices
    """
    Tx, Dx = x.shape
    Ty, Dy = y.shape
    assert Dx == Dy

    if pad:
        T = max(Tx, Ty)
        if ensure_even:
            T = T + 1 if T % 2 != 0 else T
    else:
        T = min(Tx, Ty)
        if ensure_even:
            T = T - 1 if T % 2 != 0 else T

    if Tx != T:
        if Tx < T:
            x = np.vstack(
                (x, np.zeros((T - Tx, x.shape[-1]), dtype=x.dtype)))
        else:
            x = x[:T]

    if Ty != T:
        if Ty < T:
            y = np.vstack(
                (y, np.zeros((T - Ty, y.shape[-1]), dtype=y.dtype)))
        else:
            y = y[:T]

    return x, y


def meanvar(dataset, lengths=None):
    """Mean/variance computation given a iterable dataset

    Dataset can have variable length samples. In that cases, you need to
    explicitly specify lengths for all the samples.
    """
    dtype = dataset[0].dtype

    mean_, var_ = 0., 0.
    last_sample_count = 0
    for idx, x in enumerate(dataset):
        if lengths is not None:
            x = x[:lengths[idx]]
        mean_, var_, _ = _incremental_mean_and_var(
            x, mean_, var_, last_sample_count)
        last_sample_count += len(x)
    return mean_.astype(dtype), var_.astype(dtype)


def meanstd(dataset, lengths=None):
    """Mean/std-deviation computation given a iterable dataset
    """
    m, v = meanvar(dataset, lengths)
    return m, _handle_zeros_in_scale(np.sqrt(v))


def minmax(dataset, lengths=None):
    """Min/max computation given a iterable dataset
    """
    max_ = -np.inf
    min_ = np.inf

    for idx, x in enumerate(dataset):
        if lengths is not None:
            x = x[:lengths[idx]]
        min_ = np.minimum(min_, np.min(x, axis=(0,)))
        max_ = np.maximum(max_, np.max(x, axis=(0,)))

    return min_, max_

def scale(x, data_mean, data_std):
    """Mean/variance scaling
    """
    return (x - data_mean) / _handle_zeros_in_scale(data_std, copy=False)

def minmax_scale(x, data_min, data_max, feature_range=(0, 1)):
    """Min/max scaling for given a single data.

    TODO:
        min'/scale instead of min/max?
    """
    data_range = data_max - data_min
    scale = (feature_range[1] - feature_range[0]) / \
        _handle_zeros_in_scale(data_range, copy=False)
    min_ = feature_range[0] - data_min * scale
    return x * scale + min_
