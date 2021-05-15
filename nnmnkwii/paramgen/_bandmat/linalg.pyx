# cython: language_level=3

"""Linear algebra operations for banded matrices.

Several of the functions in this module deal with symmetric banded matrices.
Symmetric banded matrices can in principle be stored even more efficiently than
general banded matrices, by not storing the superdiagonals (or subdiagonals).
However in bandmat the simpler, more explicit representation which stores the
whole band is used instead.
This is slightly less efficient in terms of memory and time, but results in a
more unified interface and simpler code in the author's limited experience.
Note that this is in line with the convention used by numpy and scipy for
symmetric "full" (i.e. non-banded) matrices, where both the upper and lower
triangles are represented in memory.
"""

# Copyright 2013, 2014, 2015, 2016, 2017, 2018 Matt Shannon

# This file is part of bandmat.
# See `License` for details of license and warranty.

from nnmnkwii.paramgen import _bandmat as bm
from nnmnkwii.paramgen._bandmat import BandMat
from nnmnkwii.paramgen._bandmat import full as fl

import numpy as np
import scipy.linalg as sla

cimport numpy as cnp
cimport cython
from libc.math cimport sqrt

cnp.import_array()
cnp.import_ufunc()

@cython.boundscheck(False)
@cython.cdivision(True)
def _cholesky_banded(cnp.ndarray[cnp.float64_t, ndim=2] mat,
                     long overwrite_ab=False,
                     long lower=False):
    """A cython reimplementation of scipy.linalg.cholesky_banded.

    Using lower == True is slightly more time and space efficient than using
    lower == False for this implementation.
    The default is False only for compatibility with the scipy implementation.

    (This simple implementation seems to be much faster than the scipy
    implementation, especially for small bandwidths.
    This is surprising since the scipy implementation wraps an LAPACK call
    which one would expect to be very fast.
    I am still unsure why this is, though I have checked it is not to do with
    Fortran-contiguous versus C-contiguous ordering.
    The scipy implementation performs a relatively expensive finite-value check
    which is arguably not necessary for our implementation, but this only
    accounts for part of the difference.)
    """
    cdef unsigned long frames, depth
    cdef cnp.ndarray[cnp.float64_t, ndim=2] mat_orig

    frames = mat.shape[1]
    assert mat.shape[0] >= 1
    depth = mat.shape[0] - 1

    mat_orig = mat
    if not lower:
        mat = fl.band_cTe(0, depth, mat)
    elif not overwrite_ab:
        mat = mat.copy()

    cdef unsigned long frame, k, l
    cdef double v0, iv0, siv0
    cdef cnp.ndarray[cnp.float64_t, ndim=1] v = np.empty(
        (depth,),
        dtype=np.float64
    )

    for frame in range(frames):
        v0 = mat[<unsigned long>(0), frame]
        if v0 <= 0.0:
            raise sla.LinAlgError(
                '%d-th leading minor not positive definite' % (frame + 1)
            )
        iv0 = 1.0 / v0
        siv0 = sqrt(iv0)

        for k in range(depth):
            v[k] = mat[k + 1, frame]

        mat[<unsigned long>(0), frame] = 1.0 / siv0
        for k in range(depth):
            mat[k + 1, frame] = v[k] * siv0

        for k in range(min(depth, frames - frame - 1)):
            for l in range(depth - k):
                mat[l, k + frame + 1] -= v[l + k] * v[k] * iv0

    if not lower:
        if overwrite_ab:
            fl.band_cTe(depth, 0, mat, target_rect=mat_orig)
            mat = mat_orig
        else:
            mat = fl.band_cTe(depth, 0, mat)

    return mat

@cython.boundscheck(False)
@cython.cdivision(True)
def _solve_triangular_banded(cnp.ndarray[cnp.float64_t, ndim=2] a_rect,
                             cnp.ndarray[cnp.float64_t, ndim=1] b,
                             long transposed=False,
                             long lower=False,
                             long overwrite_b=False):
    """A cython implementation of missing scipy.linalg.solve_triangular_banded.

    Solves A . x = b (transposed == False) or A.T . x = b (transposed == True)
    for x, where A is a upper or lower triangular banded matrix, x and b are
    vectors, . indicates matrix multiplication, and A.T denotes the transpose
    of A.
    The argument `lower` indicates whether `a_rect` stores a lower triangle or
    an upper triangle.

    This replicates functionality present in the LAPACK routine dtbtrs.
    If this function existed in scipy, it would probably be
    scipy.linalg.solve_triangular_banded.
    """
    cdef unsigned long frames, depth
    cdef long solve_lower
    cdef cnp.ndarray[cnp.float64_t, ndim=1] x

    frames = a_rect.shape[1]
    assert a_rect.shape[0] >= 1
    depth = a_rect.shape[0] - 1
    assert b.shape[0] == frames

    solve_lower = (lower != transposed)

    if overwrite_b:
        x = b
    else:
        x = np.empty_like(b)

    cdef unsigned long pos, frame, framePrev, k
    cdef double diff, denom

    for pos in range(frames):
        frame = pos if solve_lower else frames - 1 - pos
        diff = b[frame]
        if lower:
            if transposed:
                for k in range(1, min(depth + 1, pos + 1)):
                    framePrev = frame + k
                    diff -= a_rect[k, frame] * x[framePrev]
                denom = a_rect[<unsigned long>(0), frame]
            else:
                for k in range(1, min(depth + 1, pos + 1)):
                    framePrev = frame - k
                    diff -= a_rect[k, framePrev] * x[framePrev]
                denom = a_rect[<unsigned long>(0), frame]
        else:
            if transposed:
                for k in range(1, min(depth + 1, pos + 1)):
                    framePrev = frame - k
                    diff -= a_rect[depth - k, frame] * x[framePrev]
                denom = a_rect[depth, frame]
            else:
                for k in range(1, min(depth + 1, pos + 1)):
                    framePrev = frame + k
                    diff -= a_rect[depth - k, framePrev] * x[framePrev]
                denom = a_rect[depth, frame]
        if denom == 0.0:
            raise sla.LinAlgError(
                'singular matrix: resolution failed at diagonal %d' % frame
            )
        x[frame] = diff / denom

    return x

def cholesky(mat_bm, lower=False, alternative=False):
    """Computes the Cholesky factor of a positive definite banded matrix.

    The conventional Cholesky decomposition of a positive definite matrix P is
    P = L . L.T, where the lower Cholesky factor L is lower-triangular, the
    upper Cholesky factor L.T is upper-triangular, and . indicates matrix
    multiplication.
    Each positive definite matrix has a unique Cholesky decomposition.
    Given a positive definite banded matrix, this function computes its
    Cholesky factor, which is also a banded matrix.

    This function can also work with the alternative Cholesky decomposition.
    The alternative Cholesky decomposition of P is P = L.T . L, where again L
    is lower-triangular.
    Whereas the conventional Cholesky decomposition is intimately connected
    with linear Gaussian autoregressive probabilistic models defined forwards
    in time, the alternative Cholesky decomposition is intimately connected
    with linear Gaussian autoregressive probabilistic models defined backwards
    in time.

    `mat_bm` (representing P above) should represent the whole matrix, not just
    the lower or upper triangles.
    If the matrix represented by `mat_bm` is not symmetric then the behavior of
    this function is undefined (currently either the upper or lower triangle is
    used to compute the Cholesky factor and the rest of the matrix is ignored).
    If `lower` is True then the lower Cholesky factor is returned, and
    otherwise the upper Cholesky factor is returned.
    If `alternative` is True then the Cholesky factor returned is for the
    alternative Cholesky decomposition, and otherwise it is for the
    conventional Cholesky decomposition.
    """
    if mat_bm.transposed:
        mat_bm = mat_bm.T

    depth = mat_bm.l
    assert mat_bm.u == depth
    assert depth >= 0

    l = depth if lower else 0
    u = 0 if lower else depth

    mat_half_data = mat_bm.data[(depth - u):(depth + l + 1)]
    if alternative:
        chol_data = _cholesky_banded(mat_half_data[::-1, ::-1],
                                     lower=(not lower))[::-1, ::-1]
    else:
        chol_data = _cholesky_banded(mat_half_data, lower=lower)
    chol_bm = BandMat(l, u, chol_data)

    return chol_bm

def solve_triangular(a_bm, b):
    """Solves a triangular banded matrix equation.

    Solves A . x = b for x, where A is a upper or lower triangular banded
    matrix, x and b are vectors, and . indicates matrix multiplication.
    """
    assert a_bm.l == 0 or a_bm.u == 0

    transposed = a_bm.transposed
    lower = (a_bm.u == 0)
    # whether a_bm.data represents a lower or upper triangular matrix
    data_lower = (lower != transposed)
    x = _solve_triangular_banded(a_bm.data, b,
                                 transposed=transposed,
                                 lower=data_lower)
    return x

def cho_solve(chol_bm, b):
    """Solves a matrix equation given the Cholesky decomposition of the matrix.

    Solves A . x = b for x, where A is a positive definite banded matrix, x and
    b are vectors, and . indicates matrix multiplication.
    `chol_bm` is a Cholesky factor of A (either upper or lower).
    """
    assert chol_bm.l == 0 or chol_bm.u == 0

    lower = (chol_bm.u == 0)
    chol_lower_bm = chol_bm if lower else chol_bm.T

    x = solve_triangular(
        chol_lower_bm.T,
        solve_triangular(chol_lower_bm, b)
    )
    return x

def solve(a_bm, b):
    """Solves a banded matrix equation.

    Solves A . x = b for x, where A is a square banded matrix, x and b are
    vectors, and . indicates matrix multiplication.
    """
    assert a_bm.size == len(b)

    # below is necessary since sla.solve_banded does not have a transpose flag,
    #   and so is not capable of working with the transposed matrix directly.
    #   (In fact (surprisingly!) the underlying LAPACK function dgbsv does not
    #   have a transpose flag either, though gbsv calls dgbtrf and dgbtrs, and
    #   dgbtrs does have a transpose flag, so LAPACK is capable of working with
    #   the transposed matrix directly in principle).
    if a_bm.transposed:
        a_bm = a_bm.equiv(transposed_new=False)

    if a_bm.size == 0:
        x = np.zeros_like(b)
    elif a_bm.size == 1:
        # workaround for https://github.com/scipy/scipy/issues/8906
        x = b / a_bm.data[a_bm.u, 0]
    else:
        x = sla.solve_banded((a_bm.l, a_bm.u), a_bm.data, b)
    return x

def solveh(a_bm, b):
    """Solves a positive definite matrix equation.

    Solves A . x = b for x, where A is a positive definite banded matrix, x and
    b are vectors, and . indicates matrix multiplication.

    `a_bm` (representing A above) should represent the whole matrix, not just
    the lower or upper triangles.
    If the matrix represented by `a_bm` is not symmetric then the behavior of
    this function is undefined (currently either the upper or lower triangle is
    used and the rest of the matrix is ignored).
    """
    chol_bm = cholesky(a_bm, lower=True)
    x = cho_solve(chol_bm, b)
    return x

@cython.boundscheck(False)
def band_of_inverse_from_chol(chol_bm):
    """Computes the band of the inverse of a positive definite banded matrix.

    Computes the band of the inverse of a positive definite banded matrix given
    its Cholesky factor.
    Equivalently, finds the band of the covariance matrix of a discrete time
    process which is linear-Gaussian backwards in time.

    `chol_bm` can be either the lower or upper Cholesky factor.
    """
    assert chol_bm.l == 0 or chol_bm.u == 0

    if chol_bm.u != 0:
        chol_bm = chol_bm.T

    cdef unsigned long depth
    cdef long frames
    cdef long l_chol, u_chol, transposed_chol
    cdef cnp.ndarray[cnp.float64_t, ndim=2] chol_data
    cdef cnp.ndarray[cnp.float64_t, ndim=2] cov_data

    l_chol = chol_bm.l
    u_chol = chol_bm.u
    chol_data = chol_bm.data
    transposed_chol = chol_bm.transposed
    assert l_chol >= 0
    assert u_chol == 0
    assert chol_data.shape[0] == l_chol + u_chol + 1

    depth = l_chol
    frames = chol_data.shape[1]
    cov_data = np.zeros((depth * 2 + 1, frames))

    cdef long frame_l
    cdef unsigned long frame
    cdef long curr_depth
    cdef long k_1, k_2
    cdef unsigned long row_chol
    cdef long d_chol
    cdef double mult

    curr_depth = 0
    for frame_l in range(frames - 1, -1, -1):
        frame = frame_l
        row_chol = l_chol if transposed_chol else u_chol
        mult = 1.0 / chol_data[row_chol, frame]
        cov_data[depth, frame] = mult * mult
        for k_2 in range(curr_depth, -1, -1):
            for k_1 in range(curr_depth, 0, -1):
                row_chol = l_chol - k_1 if transposed_chol else u_chol + k_1
                d_chol = k_1 if transposed_chol else 0
                cov_data[depth + k_2, frame] -= (
                    chol_data[row_chol, frame + d_chol] *
                    cov_data[depth + k_1 - k_2, frame + k_2] *
                    mult
                )
        for k_2 in range(curr_depth, 0, -1):
            cov_data[depth - k_2, frame + k_2] = (
                cov_data[depth + k_2, frame]
            )
        if curr_depth < depth:
            curr_depth += 1

    cov_bm = BandMat(depth, depth, cov_data)
    return cov_bm

def band_of_inverse(mat_bm):
    """Computes band of the inverse of a positive definite banded matrix."""
    depth = mat_bm.l
    assert mat_bm.u == depth
    chol_bm = cholesky(mat_bm, lower=True)
    band_of_inv_bm = band_of_inverse_from_chol(chol_bm)
    return band_of_inv_bm
