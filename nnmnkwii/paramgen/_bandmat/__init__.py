"""A banded matrix library for python.

This package provides a simple banded matrix library for python.
It supports banded matrix-vector and matrix-matrix multiplication, converting
between full and banded matrix representations, and certain linear algebra
operations on banded matrices.

Overview
--------

A banded matrix is a matrix where only the diagonal, a number of superdiagonals
and a number of subdiagonals are non-zero.
The well-known BLAS interface and LAPACK library for linear algebra define
several banded matrix operations, and some of these, such as banded Cholesky
decomposition, are wrapped in the excellent python package scipy, specifically
in scipy.linalg.
The bandmat package re-uses the banded matrix representation used by BLAS,
LAPACK and scipy.linalg, wrapping it in a lightweight class for ease of use.
See the docstring for the BandMat class for full details of the representation
used.

The bandmat package provides:

- a lightweight class wrapping the LAPACK-style banded matrix representation.
  This class keeps track of things like bandwidths to allow a more direct
  coding style when working with banded matrices.
- some basic banded matrix operations not present in scipy.
  For example, banded matrix-vector multiplication is defined by BLAS but not
  wrapped by scipy, and banded matrix-matrix multiplication is not defined in
  BLAS or in scipy.
  The bandmat package contains C implementations of these operations written in
  cython.
- helper functions for converting between full and banded matrix
  representations.
- certain linear algebra operations on banded matrices.
- an implementation of a fancy indexed += function for numpy arrays.
  This is included for (the author's) convenience and is not directly related
  to banded matrix manipulation.

Only square banded matrices are supported by this package.

Examples
--------

Define a banded matrix:
>>> import numpy as np
>>> import bandmat as bm
>>> # Ensure consistent formatting between old and new versions of numpy.
>>> try:
...   np.set_printoptions(legacy='1.13')
... except TypeError:
...   pass
>>> a_bm = bm.BandMat(1, 1, np.arange(15, dtype=np.float64).reshape((3, 5)))
>>> print(a_bm.data)
[[  0.   1.   2.   3.   4.]
 [  5.   6.   7.   8.   9.]
 [ 10.  11.  12.  13.  14.]]

Get the equivalent numpy array:
>>> print(a_bm.full())
[[  5.   1.   0.   0.   0.]
 [ 10.   6.   2.   0.   0.]
 [  0.  11.   7.   3.   0.]
 [  0.   0.  12.   8.   4.]
 [  0.   0.   0.  13.   9.]]

Take the transpose:
>>> print(a_bm.T.full())
[[  5.  10.   0.   0.   0.]
 [  1.   6.  11.   0.   0.]
 [  0.   2.   7.  12.   0.]
 [  0.   0.   3.   8.  13.]
 [  0.   0.   0.   4.   9.]]

Banded matrix addition:
>>> b_bm = bm.BandMat(0, 1, np.arange(10, dtype=np.float64).reshape((2, 5)))
>>> print((a_bm + b_bm).full())
[[ 10.   2.   0.   0.   0.]
 [ 10.  12.   4.   0.   0.]
 [  0.  11.  14.   6.   0.]
 [  0.   0.  12.  16.   8.]
 [  0.   0.   0.  13.  18.]]

Banded matrix multiplication:
>>> b_bm = bm.BandMat(0, 1, np.arange(10, dtype=np.float64).reshape((2, 5)))
>>> print(bm.dot_mm(a_bm, b_bm).full())
[[  25.   11.    2.    0.    0.]
 [  50.   46.   26.    6.    0.]
 [   0.   66.   71.   45.   12.]
 [   0.    0.   84.  100.   68.]
 [   0.    0.    0.  104.  133.]]
"""

# Copyright 2013, 2014, 2015, 2016, 2017, 2018 Matt Shannon

# This file is part of bandmat.
# See `License` for details of license and warranty.

from .core import *  # noqa
from .tensor import *  # noqa
