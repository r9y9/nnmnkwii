"""
Utils
=====

.. autosummary::
   :toctree: generated/

   apply_delta_windows
   trim_zeros_frames
   remove_zeros_frames
   adjast_frame_length
"""

from __future__ import division, print_function, absolute_import

import numpy as np


def delta(x, win):
    return np.correlate(x, win, mode="same")


def dimention_wise_delta(x, win):
    T, D = x.shape
    y = np.zeros_like(x)
    for d in range(D):
        y[:, d] = delta(x[:, d], win)
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
        >>> from nnmnkwii.utils import apply_delta_windows
        >>> windows = [
            (0, 0, np.array(1.0)),             # static
            (1, 1, np.array(-1.0, 0, 1.0)),    # delta
            (1, 1, np.array([1.0, -2.0, 1.0])) # delta of delta
            ]
        >>> T, static_dim = 10, 24
        >>> x = np.random.rand(T, static_dim)
        >>> y = apply_delta_windows(x, windows)
        >>> assert y.shape == (T, static_dim * len(windows))
    """
    T, D = x.shape
    assert len(windows) > 0
    combined_features = np.empty((T, D * len(windows)), dtype=x.dtype)
    for idx, (_, _, window) in enumerate(windows):
        combined_features[:, D * idx:D * idx +
                          D] = dimention_wise_delta(x, window)
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
