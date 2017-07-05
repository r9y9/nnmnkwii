from __future__ import division, print_function, absolute_import

import numpy as np


def delta(x, win):
    return np.convolve(x, win, mode="same")


def dimention_wise_delta(x, win):
    T, D = x.shape
    y = np.zeros_like(x)
    for d in range(D):
        y[:, d] = delta(x[:, d], win)
    return y


def trim_zeros_frames(x):
    T, D = x.shape
    s = np.sum(x, axis=1)
    T_trimed = np.sum([s > 0])
    assert T_trimed <= T
    return x[:T_trimed]
