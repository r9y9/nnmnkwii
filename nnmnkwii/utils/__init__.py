from __future__ import division, print_function, absolute_import

import numpy as np


def trim_zeros_frames(x):
    T, D = x.shape
    s = np.sum(x, axis=1)
    T_trimed = np.sum([s > 0])
    assert T_trimed <= T
    return x[:T_trimed]
