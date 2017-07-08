from __future__ import with_statement, print_function, absolute_import

import numpy as np

import bandmat as bm
import bandmat.linalg as bla


# https://github.com/MattShannon/bandmat/blob/master/example_spg.py
# Copied from the above link. Thanks to Matt shannon!
def build_win_mats(windows, frames):
    """Builds a window matrix of a given size for each window in a collection.

    `windows` specifies the collection of windows as a sequence of
    `(l, u, win_coeff)` triples, where `l` and `u` are non-negative integers
    specifying the left and right extents of the window and `win_coeff` is an
    array specifying the window coefficients.
    The returned value is a list of window matrices, one for each of the
    windows specified in `windows`.
    Each window matrix is a `frames` by `frames` Toeplitz matrix with lower
    bandwidth `l` and upper bandwidth `u`.
    The non-zero coefficients in each row of this Toeplitz matrix are given by
    `win_coeff`.
    The returned window matrices are stored as BandMats, i.e. using a banded
    representation.
    """
    win_mats = []
    for l, u, win_coeff in windows:
        assert l >= 0 and u >= 0
        assert len(win_coeff) == l + u + 1
        win_coeffs = np.tile(np.reshape(win_coeff, (l + u + 1, 1)), frames)
        win_mat = bm.band_c_bm(u, l, win_coeffs).T
        win_mats.append(win_mat)

    return win_mats


def build_poe(b_frames, tau_frames, win_mats, sdw=None):
    r"""Computes natural parameters for a Gaussian product-of-experts model.

    The natural parameters (b-value vector and precision matrix) are returned.
    The returned precision matrix is stored as a BandMat.
    Mathematically the b-value vector is given as:

        b = \sum_d \transpose{W_d} \tilde{b}_d

    and the precision matrix is given as:

        P = \sum_d \transpose{W_d} \text{diag}(\tilde{tau}_d) W_d

    where $W_d$ is the window matrix for window $d$ as specified by an element
    of `win_mats`, $\tilde{b}_d$ is the sequence over time of b-value
    parameters for window $d$ as given by a column of `b_frames`, and
    $\tilde{\tau}_d$ is the sequence over time of precision parameters for
    window $d$ as given by a column of `tau_frames`.
    """
    if sdw is None:
        sdw = max([win_mat.l + win_mat.u for win_mat in win_mats])
    num_windows = len(win_mats)
    frames = len(b_frames)
    assert np.shape(b_frames) == (frames, num_windows)
    assert np.shape(tau_frames) == (frames, num_windows)
    assert all([win_mat.l + win_mat.u <= sdw for win_mat in win_mats])

    b = np.zeros((frames,))
    prec = bm.zeros(sdw, sdw, frames)

    for win_index, win_mat in enumerate(win_mats):
        bm.dot_mv_plus_equals(win_mat.T, b_frames[:, win_index], target=b)
        bm.dot_mm_plus_equals(win_mat.T, win_mat, target_bm=prec,
                              diag=tau_frames[:, win_index])

    return b, prec


def mlpg_numpy(mean_frames, variance_frames, windows):
    """Numpy implementation of MLPG
    """
    T, D = mean_frames.shape
    assert mean_frames.shape == variance_frames.shape
    static_dim = D // len(windows)

    num_windows = len(windows)
    win_mats = build_win_mats(windows, T)

    # workspaces; those will be updated in the following generation loop
    means = np.zeros((T, num_windows))
    precisions = np.zeros((T, num_windows))
    bs = np.zeros((T, num_windows))

    # Perform dimention-wise generation
    y = np.zeros((T, static_dim))
    for d in range(static_dim):
        for win_idx in range(num_windows):
            means[:, win_idx] = mean_frames[:, win_idx * static_dim + d]
            precisions[:, win_idx] = 1 / \
                variance_frames[:, win_idx * static_dim + d]
            bs[:, win_idx] = precisions[:, win_idx] * means[:, win_idx]

        b, P = build_poe(bs, precisions, win_mats)
        y[:, d] = bla.solveh(P, b)

    return y
