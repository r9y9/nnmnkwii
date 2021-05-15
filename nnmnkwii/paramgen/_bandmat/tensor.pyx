# cython: language_level=3

"""Multiplication, etc using banded matrices."""

# Copyright 2013, 2014, 2015, 2016, 2017, 2018 Matt Shannon

# This file is part of bandmat.
# See `License` for details of license and warranty.

from nnmnkwii.paramgen._bandmat import core as bm_core

import numpy as np

cimport numpy as cnp
cimport cython

cnp.import_array()
cnp.import_ufunc()

@cython.boundscheck(False)
def dot_mv_plus_equals(a_bm,
                       cnp.ndarray[cnp.float64_t, ndim=1] b,
                       cnp.ndarray[cnp.float64_t, ndim=1] target):
    """Multiplies a banded matrix by a vector, adding the result to a vector.

    The statement `dot_mv_plus_equals(a_bm, b, target)` where `a_bm` is a
    BandMat is the equivalent of:

        target += np.dot(a_full, b)

    where `a_full` is a square numpy array.
    """
    # (FIXME : could wrap corresponding BLAS routine (gbmv) instead)
    cdef long frames
    cdef long l_a, u_a, transposed
    cdef cnp.ndarray[cnp.float64_t, ndim=2] a_data

    l_a = a_bm.l
    u_a = a_bm.u
    a_data = a_bm.data
    transposed = a_bm.transposed
    assert l_a >= 0
    assert u_a >= 0
    assert a_data.shape[0] == l_a + u_a + 1

    frames = a_data.shape[1]
    assert b.shape[0] == frames
    assert target.shape[0] == frames

    cdef long o_a
    cdef unsigned long row_a
    cdef long d_a
    cdef unsigned long frame

    for o_a in range(-u_a, l_a + 1):
        row_a = (l_a - o_a) if transposed else (u_a + o_a)
        d_a = 0 if transposed else -o_a
        for frame in range(max(0, o_a), max(0, frames + min(0, o_a))):
            target[frame] += (
                a_data[row_a, frame + d_a] *
                b[frame - o_a]
            )

    return

def dot_mv(a_bm, b):
    """Multiplies a banded matrix by a vector.

    The expression `dot_mv(a_bm, b)` where `a_bm` is a BandMat is the
    equivalent of:

        np.dot(a_full, b)

    where `a_full` is a square numpy array.
    """
    size = len(b)
    assert a_bm.size == size
    c = np.zeros((size,))
    dot_mv_plus_equals(a_bm, b, c)
    return c

@cython.boundscheck(False)
def dot_mm_plus_equals(a_bm, b_bm, target_bm,
                       cnp.ndarray[cnp.float64_t, ndim=1] diag=None):
    """Multiplies two banded matrices, adding the result to a banded matrix.

    If `diag` is None, computes A . B, where . indicates matrix multiplication,
    and adds the result to `target_bm`.
    If `diag` is not None, computes A . D . B, where D is the diagonal matrix
    with diagonal `diag`, and adds result to `target_bm`.

    If `target_bm` does not contain enough rows to contain the result of A . B,
    then only diagonals of A . B which contribute to the rows present in the
    banded representation used by `target_bm` are computed.
    This is more efficient in the case that not all diagonals of `target_bm`
    are needed.

    The statement `dot_mm_plus_equals(a_bm, b_bm, target_bm, diag=None)` where
    `a_bm`, `b_bm` and `target_bm` are BandMats is the equivalent of:

        target_full += band_ec(l, u, np.dot(a_full, b_full))

    where `a_full`, `b_full` and `target_full` are square numpy arrays.
    Here `l` is `target_bm.l` and `u` is `target_bm.u`.
    If `diag` is not None then it is the equivalent of:

        target_full += (
            band_ec(l, u, np.dot(np.dot(a_full, np.diag(diag)), b_full))
        )
    """
    cdef long frames
    cdef long l_a, u_a, transposed_a
    cdef long l_b, u_b, transposed_b
    cdef long l_c, u_c, transposed_c
    cdef long use_diag
    cdef cnp.ndarray[cnp.float64_t, ndim=2] a_data
    cdef cnp.ndarray[cnp.float64_t, ndim=2] b_data
    cdef cnp.ndarray[cnp.float64_t, ndim=2] c_data

    l_a = a_bm.l
    u_a = a_bm.u
    a_data = a_bm.data
    transposed_a = a_bm.transposed
    assert l_a >= 0
    assert u_a >= 0
    assert a_data.shape[0] == l_a + u_a + 1

    l_b = b_bm.l
    u_b = b_bm.u
    b_data = b_bm.data
    transposed_b = b_bm.transposed
    assert l_b >= 0
    assert u_b >= 0
    assert b_data.shape[0] == l_b + u_b + 1

    l_c = target_bm.l
    u_c = target_bm.u
    c_data = target_bm.data
    transposed_c = target_bm.transposed
    assert l_c >= 0
    assert u_c >= 0
    assert c_data.shape[0] == l_c + u_c + 1

    frames = a_data.shape[1]
    assert b_data.shape[1] == frames
    assert c_data.shape[1] == frames

    use_diag = (diag is not None)
    if use_diag:
        assert diag.shape[0] == frames

    cdef long o_a, o_b, o_c
    cdef unsigned long row_a, row_b, row_c
    cdef long d_a, d_b, d_c
    cdef unsigned long frame

    for o_c in range(-min(u_c, u_a + u_b), min(l_c, l_a + l_b) + 1):
        for o_a in range(-min(u_a, l_b - o_c), min(l_a, u_b + o_c) + 1):
            o_b = o_c - o_a
            row_a = (l_a - o_a) if transposed_a else (u_a + o_a)
            row_b = (l_b - o_b) if transposed_b else (u_b + o_b)
            row_c = (l_c - o_c) if transposed_c else (u_c + o_c)
            d_a = o_a if transposed_a else 0
            d_b = 0 if transposed_b else -o_b
            d_c = o_a if transposed_c else -o_b
            for frame in range(max(0, -o_a, o_b),
                               max(0, frames + min(0, -o_a, o_b))):
                c_data[row_c, frame + d_c] += (
                    a_data[row_a, frame + d_a] *
                    b_data[row_b, frame + d_b] *
                    (diag[frame] if use_diag else 1.0)
                )

    return

def dot_mm(a_bm, b_bm, diag=None):
    """Multiplies two banded matrices.

    If `diag` is None, computes A . B, where . indicates matrix multiplication.
    If `diag` is not None, computes A . D . B, where D is the diagonal matrix
    with diagonal `diag`.

    The expression `dot_mm(a_bm, b_bm, diag=None)` where `a_bm` and `b_bm` are
    BandMats is the equivalent of:

        np.dot(a_full, b_full)

    where `a_full` and `b_full` are square numpy arrays.
    If `diag` is not None then it is the equivalent of:

        np.dot(np.dot(a_full, np.diag(diag)), b_full)
    """
    assert a_bm.size == b_bm.size
    c_bm = bm_core.zeros(a_bm.l + b_bm.l, a_bm.u + b_bm.u, a_bm.size)
    dot_mm_plus_equals(a_bm, b_bm, c_bm, diag=diag)
    return c_bm

def dot_mm_partial(l, u, a_bm, b_bm, diag=None):
    """Computes part of the result of multiplying two banded matrices.

    If `diag` is None, computes part of C = A . B, where . indicates matrix
    multiplication.
    If `diag` is not None, computes part of C = A . D . B, where D is the
    diagonal matrix with diagonal `diag`.

    This function only computes the diagonals of C that are within the band
    specified by `l` and `u`.
    This is desirable in the case that not all diagonals of C are needed.

    The expression `dot_mm_partial(l, u, a_bm, b_bm, diag=None)` where `a_bm`
    and `b_bm` are BandMats is the equivalent of:

        band_ec(l, u, np.dot(a_full, b_full))

    where `a_full` and `b_full` are square numpy arrays.
    If `diag` is not None then it is the equivalent of:

        band_ec(l, u, np.dot(np.dot(a_full, np.diag(diag)), b_full))
    """
    assert a_bm.size == b_bm.size
    c_bm = bm_core.zeros(l, u, a_bm.size)
    dot_mm_plus_equals(a_bm, b_bm, c_bm, diag=diag)
    return c_bm

def dot_mmm_partial(l, u, a_bm, b_bm, c_bm):
    """Computes part of the result of multiplying three banded matrices.

    Computes D = A . (B . C), where . indicates matrix multiplication.

    This function only computes the diagonals of D that are within the band
    specified by `l` and `u`.
    Furthermore the intermediate result E = B . C is also only partially
    computed, with rows that can no have influence on the final result never
    being computed.
    This behavior is desirable in the case that not all diagonals of D are
    needed.

    The expression `dot_mmm_partial(l, u, a_bm, b_bm, c_bm)` where `a_bm`,
    `b_bm` and `c_bm` are BandMats is the equivalent of:

        band_ec(l, u, np.dot(a_full, np.dot(b_full, c_full)))

    where `a_full`, `b_full` and `c_full` are square numpy arrays.
    """
    l_i = l + a_bm.u
    u_i = u + a_bm.l
    return dot_mm_partial(l, u, a_bm, dot_mm_partial(l_i, u_i, b_bm, c_bm))

@cython.boundscheck(False)
def band_of_outer_plus_equals(cnp.ndarray[cnp.float64_t, ndim=1] a_vec,
                              cnp.ndarray[cnp.float64_t, ndim=1] b_vec,
                              target_bm,
                              double mult=1.0):
    """Adds the band of the outer product of two vectors to a banded matrix.

    The statement `band_of_outer_plus_equals(a_vec, b_vec, target_bm, mult)`
    where `target_bm` is a BandMat is the equivalent of:

        target_full += band_ec(l, u, np.outer(a_vec, b_vec) * mult)

    where `target_full` is a square numpy array.
    Here `l` is `target_bm.l` and `u` is `target_bm.u`.
    """
    cdef long frames
    cdef long l_c, u_c, transposed_c
    cdef cnp.ndarray[cnp.float64_t, ndim=2] c_data

    l_c = target_bm.l
    u_c = target_bm.u
    c_data = target_bm.data
    transposed_c = target_bm.transposed
    assert l_c >= 0
    assert u_c >= 0
    assert c_data.shape[0] == l_c + u_c + 1

    frames = c_data.shape[1]
    assert a_vec.shape[0] == frames
    assert b_vec.shape[0] == frames

    cdef long o_c
    cdef unsigned long row_c
    cdef long d_c
    cdef unsigned long frame

    for o_c in range(-u_c, l_c + 1):
        row_c = (l_c - o_c) if transposed_c else (u_c + o_c)
        d_c = o_c if transposed_c else 0
        for frame in range(max(0, -o_c), max(0, frames + min(0, -o_c))):
            c_data[row_c, frame + d_c] += (
                a_vec[frame + o_c] * b_vec[frame] * mult
            )

    return

def trace_dot(a_bm, b_bm):
    """Convenience method to compute the matrix inner product.

    The expression `trace_dot(a_bm, b_bm)` where `a_bm` and `b_bm` are BandMats
    is the equivalent of:

        np.trace(np.dot(a_full.T, b_full))

    where `a_full` and `b_full` are square numpy arrays.

    This operation is a valid inner product over the vector space of square
    matrices of a given size.
    """
    return np.sum(bm_core.diag(dot_mm_partial(0, 0, a_bm.T, b_bm)))
