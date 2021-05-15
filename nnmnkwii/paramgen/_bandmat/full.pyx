# cython: language_level=3

"""Operations involving the bands of square matrices.

This module provides operations involving the bands of square matrices which
are stored as numpy arrays.
For example it provides functions to construct a square matrix with given band,
and to extract the band from an arbitrary square matrix.
Though this is closely related to the representation of square banded matrices
used by the `BandMat` class in the `core` module, it is logically distinct.
"""

# Copyright 2013, 2014, 2015, 2016, 2017, 2018 Matt Shannon

# This file is part of bandmat.
# See `License` for details of license and warranty.

import numpy as np

cimport numpy as cnp
cimport cython

cnp.import_array()
cnp.import_ufunc()

@cython.boundscheck(False)
def band_c(long l, long u, cnp.ndarray[cnp.float64_t, ndim=2] mat_rect):
    """Constructs a square banded matrix from its band.

    Given a rectangular numpy array `mat_rect`, this function returns a square
    numpy array `mat_full` with its `u` superdiagonals, diagonal and `l`
    subdiagonals given by the rows of `mat_rect`.
    The part of each column of `mat_full` that lies within the band contains
    the same entries as the corresponding column of `mat_rect`.

    Not all the entries of `mat_rect` affect `mat_full`.
    The (l, u)-extra entries of a rectangular matrix `mat_rect` are defined as
    the entries which have no effect on the result of `band_c(l, u, mat_rect)`.
    They lie in the upper-left and bottom-right corners of `mat_rect`.
    """
    assert l >= 0
    assert u >= 0
    assert mat_rect.shape[0] == l + u + 1

    cdef long size
    cdef cnp.ndarray[cnp.float64_t, ndim=2] mat_full

    size = mat_rect.shape[1]
    mat_full = np.zeros((size, size))

    cdef long i
    cdef unsigned long row
    cdef unsigned long j

    for i in range(-u, l + 1):
        row = u + i
        for j in range(max(0, -i), max(0, size + min(0, -i))):
            mat_full[j + i, j] = mat_rect[row, j]

    return mat_full

@cython.boundscheck(False)
def band_e(long l, long u, cnp.ndarray[cnp.float64_t, ndim=2] mat_full):
    """Extracts a band of a square matrix.

    Given a square numpy array `mat_full`, returns a rectangular numpy array
    `mat_rect` with rows corresponding to the `u` superdiagonals, the diagonal
    and the `l` subdiagonals of `mat_full`.
    The square matrix is "collapsed column-wise", i.e. each column of
    `mat_rect` contains the same entries as the part of the corresponding
    column of `mat_full` that lies within the band defined by (l, u).
    The extra entries in the corners of `mat_rect` which do not correspond to
    any entry in `mat_full` are set to zero.
    """
    assert l >= 0
    assert u >= 0
    assert mat_full.shape[1] == mat_full.shape[0]

    cdef long size
    cdef cnp.ndarray[cnp.float64_t, ndim=2] mat_rect

    size = mat_full.shape[0]
    mat_rect = np.empty((l + u + 1, size))

    cdef long i
    cdef unsigned long row
    cdef unsigned long j

    for i in range(-u, l + 1):
        row = u + i
        for j in range(size):
            # "j + i < size" below uses wraparound on unsigned long to actually
            #   check that "0 <= j + i < size"
            mat_rect[row, j] = mat_full[j + i, j] if j + i < size else 0.0

    return mat_rect

@cython.boundscheck(False)
def zero_extra_entries(long l, long u,
                       cnp.ndarray[cnp.float64_t, ndim=2] mat_rect):
    """Zeroes the extra entries of a rectangular matrix.

    See the docstring for `band_c` for the definition of extra entries.

    The statement `zero_extra_entries(l, u, mat_rect)` is equivalent to:

        mat_rect[:] = band_e(l, u, band_c(l, u, mat_rect))

    N.B. in-place, i.e. mutates `mat_rect`.
    """
    assert l >= 0
    assert u >= 0
    assert mat_rect.shape[0] == l + u + 1

    cdef long size

    size = mat_rect.shape[1]

    cdef long i
    cdef unsigned long row
    cdef unsigned long j

    for i in range(-u, 0):
        row = u + i
        for j in range(0, min(size, -i)):
            mat_rect[row, j] = 0.0
    for i in range(1, l + 1):
        row = u + i
        for j in range(max(0, size - i), size):
            mat_rect[row, j] = 0.0

    return

def band_ce(l, u, mat_rect):
    """Effectively applies `band_c` then `band_e`.

    The combined operation has the effect of zeroing the extra entries in (a
    copy of) the rectangular matrix `mat_rect`.

    The expression `band_ce(l, u, mat_rect)` is equivalent to:

        band_e(l, u, band_c(l, u, mat_rect))
    """
    mat_rect_new = mat_rect.copy()
    zero_extra_entries(l, u, mat_rect_new)
    return mat_rect_new

def band_ec(l, u, mat_full):
    """Effectively applies `band_e` then `band_c`.

    The combined operation has the effect of zeroing the entries outside the
    band specified by `l` and `u` in (a copy of) the square matrix `mat_full`.

    The expression `band_ec(l, u, mat_full)` is equivalent to:

        band_c(l, u, band_e(l, u, mat_full))
    """
    return band_c(l, u, band_e(l, u, mat_full))

@cython.boundscheck(False)
def band_cTe(long l, long u,
             cnp.ndarray[cnp.float64_t, ndim=2] mat_rect,
             cnp.ndarray[cnp.float64_t, ndim=2] target_rect=None):
    """Effectively applies `band_c` then transpose then `band_e`.

    The expression `band_cTe(l, u, mat_rect)` is equivalent to:

        band_e(u, l, band_c(l, u, mat_rect).T)
    """
    assert l >= 0
    assert u >= 0
    assert mat_rect.shape[0] == l + u + 1

    cdef long size
    cdef long target_given = (target_rect is not None)

    size = mat_rect.shape[1]
    if target_given:
        assert target_rect.shape[0] == l + u + 1
        assert target_rect.shape[1] == size
    else:
        target_rect = np.empty((l + u + 1, size))

    cdef long i
    cdef unsigned long row, row_new
    cdef unsigned long j

    for i in range(-u, l + 1):
        row = u + i
        row_new = l - i
        for j in range(size):
            # "j - i < size" below uses wraparound on unsigned long to actually
            #   check that "0 <= j - i < size"
            target_rect[row_new, j] = (mat_rect[row, j - i] if j - i < size
                                       else 0.0)

    if target_given:
        return
    else:
        return target_rect
