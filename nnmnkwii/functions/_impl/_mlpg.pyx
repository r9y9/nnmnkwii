# coding: utf-8
# cython: wraparound = False
# cython: boundscheck = False

import numpy as np
cimport numpy as np


def full_window_mat(win_mats, int T):
    cdef np.ndarray[np.float64_t, ndim = 2] mat_full

    mat_full = np.zeros((T * len(win_mats), T))

    cdef long size
    cdef long i, j, u, l, row
    cdef long win_index
    cdef long transposed

    for win_index, win_mat in enumerate(win_mats):
        transposed = win_mat.transposed
        row_offset = win_index * T
        u = win_mat.u
        l = win_mat.l
        mat_rect = win_mat.data
        size = mat_rect.shape[1]
        for i in range(-u, l + 1):
            row = l - i if transposed else u + i
            for j in range(max(0, -i), max(0, size + min(0, -i))):
                mat_full[row_offset + j + i, j] = mat_rect[row, j]

    return mat_full
