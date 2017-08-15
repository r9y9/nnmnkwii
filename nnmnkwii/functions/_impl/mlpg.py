# coding: utf-8
from __future__ import with_statement, print_function, absolute_import

import numpy as np

import bandmat as bm
import bandmat.linalg as bla
from scipy.linalg import solve_banded

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


def mlpg(mean_frames, variance_frames, windows):
    """Maximum Parameter Likelihood Generation (MLPG)

    Function ``f: (T, D) -> (T, static_dim)``.

    It peforms Maximum Likelihood Parameter Generation (MLPG) algorithm
    to generate static features from static + dynamic features over
    time frames. The implementation was heavily inspired by [1]_ and
    using bandmat_ for efficient computation.

    .. _bandmat: https://github.com/MattShannon/bandmat

    .. [1] M. Shannon, supervised by W. Byrne (2014),
      Probabilistic acoustic modelling for parametric speech synthesis
      PhD thesis, University of Cambridge, UK

    Args:
        mean_frames (2darray): The input features (static + delta).
            In statistical speech synthesis, these are means of gaussian
            distributions predicted by neural networks or decision trees.
        variance_frames (2d or 1darray): Variances (static + delta ) of gaussian
            distributions over time frames (2d) or global variances (1d).
            If global variances are given, these will get expanded over frames.
        windows (list): A sequence of ``(l, u, win_coeff)`` triples, where
            ``l`` and ``u`` are non-negative integers specifying the left
            and right extents of the window and `win_coeff` is an array
            specifying the window coefficients.

    Returns:
        Generated static features over time

    Examples:
        >>> from nnmnkwii import functions as F
        >>> windows = [
        ...         (0, 0, np.array([1.0])),            # static
        ...         (1, 1, np.array([-0.5, 0.0, 0.5])), # delta
        ...         (1, 1, np.array([1.0, -2.0, 1.0])), # delta-delta
        ...     ]
        >>> T, static_dim = 10, 24
        >>> mean_frames = np.random.rand(T, static_dim * len(windows))
        >>> variance_frames = np.random.rand(T, static_dim * len(windows))
        >>> static_features = F.mlpg(mean_frames, variance_frames, windows)
        >>> assert static_features.shape == (T, static_dim)

    See also:
        :func:`nnmnkwii.autograd.mlpg`

    """
    T, D = mean_frames.shape
    # expand variances over frames
    if variance_frames.ndim == 1 and variance_frames.shape[0] == D:
        variance_frames = np.tile(variance_frames, (T, 1))
    assert mean_frames.shape == variance_frames.shape
    static_dim = D // len(windows)

    num_windows = len(windows)
    win_mats = build_win_mats(windows, T)

    # workspaces; those will be updated in the following generation loop
    means = np.zeros((T, num_windows))
    precisions = np.zeros((T, num_windows))
    # Perform dimention-wise generation
    y = np.zeros((T, static_dim))
    for d in range(static_dim):

        for win_idx in range(num_windows):
            means[:, win_idx] = mean_frames[:, win_idx * static_dim + d]
            precisions[:, win_idx] = 1 / \
                variance_frames[:, win_idx * static_dim + d]

        bs = precisions * means
        b, P = build_poe(bs, precisions, win_mats)
        y[:, d] = bla.solveh(P, b)

    return y


def mlpg_grad(mean_frames, variance_frames, windows, grad_output):
    """MLPG gradient computation

    Parameters are almost same as :func:`nnmnkwii.functions.mlpg`. See the
    function docmenent for what the parameters mean.

    Args:
        mean_frames (numpy.ndarray): Means.
        variance_frames (numpy.ndarray): Variances.
        windows (list): Windows.
        grad_output: Backpropagated output gradient, shape (``T x static_dim``)

    Returns:
        numpy.ndarray: Gradients to be back propagated, shape: (``T x D``)

    See also:
        :func:`nnmnkwii.autograd.mlpg`, :class:`nnmnkwii.autograd.MLPG`
    """
    T, D = mean_frames.shape
    win_mats = build_win_mats(windows, T)
    static_dim = D // len(windows)

    grads = np.zeros((T, D), dtype=np.float32)
    for d in range(static_dim):
        sdw = max([win_mat.l + win_mat.u for win_mat in win_mats])

        # R: \sum_{l} W_{l}^{T}P_{d,l}W_{l}
        R = bm.zeros(sdw, sdw, T)  # overwritten in the loop

        # dtype = np.float64 for bandmat
        precisions = np.zeros((len(windows), T), dtype=np.float64)

        for win_idx, win_mat in enumerate(win_mats):
            precisions[win_idx] = 1 / \
                variance_frames[:, win_idx * static_dim + d]

            bm.dot_mm_plus_equals(win_mat.T, win_mat,
                                  target_bm=R, diag=precisions[win_idx])

        for win_idx, win_mat in enumerate(win_mats):
            # r: W_{l}^{T}P_{d,l}
            r = bm.dot_mm(win_mat.T, bm.diag(precisions[win_idx]))

            # grad_{d, l} = R^{-1r}
            grad = solve_banded((R.l, R.u), R.data, r.full())
            assert grad.shape == (T, T)

            # Finally we get grad for a particular dimention
            grads[:, win_idx * static_dim + d] = grad_output[:, d].T.dot(grad)

    return grads
