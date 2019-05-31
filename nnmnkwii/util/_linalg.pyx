# coding: utf-8
# cython: wraparound = False
# cython: boundscheck = False
# cython: language_level=3

cimport cython
from scipy.linalg.cython_lapack cimport dpotri
cimport numpy as np
import numpy as np

cdef inline void to_fullstrage_L(double[::1] A, int N, int LDA):
    cdef int row, col
    for row in range(N):
        for col in range(row + 1, LDA):
            A[col * LDA + row] = A[row * LDA + col]

cdef inline void to_fullstrage_U(double[::1] A, int N, int LDA):
    cdef int row, col
    for row in range(N):
        for col in range(row + 1, LDA):
            A[row * LDA + col] = A[col * LDA + row]


def dpotri_full_L(int N, double[::1] A, int LDA):
    cdef int info
    dpotri("L", & N, & A[0], & LDA, & info)
    to_fullstrage_L(A, N, LDA)
    return np.asarray(A)


def dpotri_full_U(int N, double[::1] A, int LDA):
    cdef int info
    dpotri("U", & N, & A[0], & LDA, & info)
    to_fullstrage_U(A, N, LDA)
    return np.asarray(A)

# Adapted from:
# https://github.com/SythonUK/whisperVC
# TODO: understand what it really does.
# I guess this is utilizing banded property of the matrix to improve
# compuational efficiencly. It's about two times faster than lapack's generic
# inverse of real positive definite matrix (dpotri).


def cholesky_inv_banded(R, int w):
    cdef int T = R.shape[0]
    g = np.zeros((T, T))
    g[0, 0] = 1.0 / R[0, 0]
    hold = np.zeros(T)
    P = np.zeros((T, T))

    cdef int t, i, j
    for t in range(1, T):
        hold *= 0.0
        for j in range(1, w):
            if (t - j >= 0) and (R[t, t - j] != 0.0):
                hold[0:t + 1] += R[t, t - j] * g[t - j, 0:t + 1]
        hold[t] -= 1.0
        g[t, 0:t + 1] = - hold[0:t + 1] / R[t, t]

    P[T - 1, :] = g[T - 1, :] / R[T - 1, T - 1]
    R = R.T

    for t in range(T - 2, -1, -1):
        hold *= 0.0
        for j in range(1, w):
            if (t + j < T) and (R[t, t + j] != 0.0):
                hold += R[t, t + j] * P[t + j, :]
        P[t, :] = (g[t, :] - hold) / R[t, t]

    return P
