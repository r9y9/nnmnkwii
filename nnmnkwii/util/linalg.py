# coding: utf-8
from __future__ import division, print_function, absolute_import

from ._linalg import dpotri_full_L, dpotri_full_U
from ._linalg import cholesky_inv_banded as _cholesky_inv_banded
import numpy as np


def cholesky_inv(L, lower=False):
    """Compute inverse matrix of real symmetric positive definite matrix
    using choleskey factorization.

    The function internally uses LAPACK's ``dportri``.

    Args:
        L (numpy.ndarray[dtype=float64]): Lower or upper triangle matrix given
          by choleskey factorization.
        lower: Set ``True` if you give lower triangle matrix, ``False``
          otherwise.

    Returns:
        numpy.ndarrray: Inverse matrix (full storage).
    """
    N1, N2 = L.shape
    assert N1 == N2
    assert L.dtype == np.float64
    N = N1

    # transpose for array order difference (probably)
    B = L.T.copy()

    # Inplace operations
    if lower:
        dpotri_full_L(N, B.ravel(), N)
    else:
        dpotri_full_U(N, B.ravel(), N)

    return B


def cholesky_inv_banded(L, width=3):
    # TODO
    return _cholesky_inv_banded(L, width)
