from __future__ import division, print_function, absolute_import

from nnmnkwii.util import apply_each2d_padded, apply_each2d_trim
from nnmnkwii.util.linalg import cholesky_inv, cholesky_inv_banded

import numpy as np
import scipy.linalg

from nose.plugins.attrib import attr


def _get_windows_set():
    windows_set = [
        # Static
        [
            (0, 0, np.array([1.0])),
        ],
        # Static + delta
        [
            (0, 0, np.array([1.0])),
            (1, 1, np.array([-0.5, 0.0, 0.5])),
        ],
        # Static + delta + deltadelta
        [
            (0, 0, np.array([1.0])),
            (1, 1, np.array([-0.5, 0.0, 0.5])),
            (1, 1, np.array([1.0, -2.0, 1.0])),
        ],
    ]
    return windows_set


def test_function_utils():
    def dummmy_func2d(x):
        return x + 1

    T, D = 10, 24
    np.random.seed(1234)
    X = np.random.rand(2, T, D)
    lengths = [60, 100]

    # Paddd case
    Y = apply_each2d_padded(dummmy_func2d, X, lengths)
    for i, l in enumerate(lengths):
        assert np.allclose(X[i][:l] + 1, Y[i][:l])
        assert np.all(Y[i][l:] == 0)

    # Trim
    for i, l in enumerate(lengths):
        X[i][l:] = 0
    Y = apply_each2d_trim(dummmy_func2d, X)
    for i, l in enumerate(lengths):
        assert np.allclose(X[i][:l] + 1, Y[i][:l])
        assert np.all(Y[i][l:] == 0)


def _get_banded_test_mat(win_mats, T):
    import bandmat as bm

    sdw = max([win_mat.l + win_mat.u for win_mat in win_mats])
    P = bm.zeros(sdw, sdw, T)
    for win_index, win_mat in enumerate(win_mats):
        bm.dot_mm_plus_equals(win_mat.T, win_mat, target_bm=P)
    return P


@attr("requires_bandmat")
def test_linalg_choleskey_inv():
    from nnmnkwii.paramgen import build_win_mats

    for windows in _get_windows_set():
        for T in [5, 10]:
            win_mats = build_win_mats(windows, T)
            P = _get_banded_test_mat(win_mats, T).full()
            L = scipy.linalg.cholesky(P, lower=True)
            U = scipy.linalg.cholesky(P, lower=False)
            assert np.allclose(L.dot(L.T), P)
            assert np.allclose(U.T.dot(U), P)

            Pinv = np.linalg.inv(P)
            Pinv_hat = cholesky_inv(L, lower=True)
            assert np.allclose(Pinv, Pinv_hat)
            Pinv_hat = cholesky_inv(U, lower=False)
            assert np.allclose(Pinv, Pinv_hat)

            Pinv_hat = cholesky_inv_banded(L, width=3)
            assert np.allclose(Pinv, Pinv_hat)
