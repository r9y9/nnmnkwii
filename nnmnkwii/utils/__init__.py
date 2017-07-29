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


def remove_zeros_frames(x):
    T, D = x.shape
    s = np.sum(x, axis=1)
    return x[s > 0]


def trim_zeros_frames(x):
    T, D = x.shape
    s = np.sum(x, axis=1)
    T_trimed = np.sum([s > 0])
    assert T_trimed <= T
    return x[:T_trimed]


def adjast_frame_length(x, y, pad=True, ensure_even=False):
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
