# coding: utf-8
from __future__ import with_statement, print_function, absolute_import

import numpy as np

import bandmat as bm
import bandmat.linalg as bla
from scipy.linalg import solve_banded
from nnmnkwii.util.linalg import cholesky_inv_banded
from .mlpg_helper import full_window_mat as _full_window_mat

# https://github.com/MattShannon/bandmat/blob/master/example_spg.py
# Copied from the above link. Thanks to Matt shannon!


def build_win_mats(windows, T):
    """Builds a window matrix of a given size for each window in a collection.

    Args:
        windows(list): specifies the collection of windows as a sequence of
        ``(l, u, win_coeff)`` triples, where ``l`` and ``u`` are non-negative
        integers pecifying the left and right extents of the window and
        ``win_coeff`` is an array specifying the window coefficients.
        T (int): Number of frames.

    Returns:
        list: The returned value is a list of window matrices, one for each of
        the windows specified in ``windows``. Each window matrix is a
        ``T`` by ``T`` Toeplitz matrix with lower bandwidth ``l`` and upper
        bandwidth ``u``. The non-zero coefficients in each row of this Toeplitz
        matrix are given by ``win_coeff``.
        The returned window matrices are stored as BandMats, i.e. using a
        banded representation.

    Examples:
        >>> from nnmnkwii import paramgen as G
        >>> import numpy as np
        >>> windows = [
        ...     (0, 0, np.array([1.0])),            # static
        ...     (1, 1, np.array([-0.5, 0.0, 0.5])), # delta
        ...     (1, 1, np.array([1.0, -2.0, 1.0])), # delta-delta
        ... ]
        >>> win_mats = G.build_win_mats(windows, 3)
    """
    win_mats = []
    for l, u, win_coeff in windows:
        assert l >= 0 and u >= 0
        assert len(win_coeff) == l + u + 1
        win_coeffs = np.tile(np.reshape(win_coeff, (l + u + 1, 1)), T)
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
    time frames dimension-by-dimension.

    Let :math:`\mu` (``T x 1``) is the input mean sequence of a particular
    dimension and :math:`y` (``T x 1``) is the static
    feature sequence we want to compute, the formula of MLPG is written as:

    .. math::

        y = A^{-1} b

    where

    .. math::

        A = \sum_{l} W_{l}^{T}P_{l}W_{l}

    ,

    .. math::

        b = P\mu

    :math:`W_{l}` is the ``l``-th window matrix (``T x T``) and :math:`P`
    (``T x T``) is the precision matrix which is given by the inverse of
    variance matrix.

    The implementation was heavily inspired by [1]_ and
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
        >>> from nnmnkwii import paramgen as G
        >>> windows = [
        ...         (0, 0, np.array([1.0])),            # static
        ...         (1, 1, np.array([-0.5, 0.0, 0.5])), # delta
        ...         (1, 1, np.array([1.0, -2.0, 1.0])), # delta-delta
        ...     ]
        >>> T, static_dim = 10, 24
        >>> mean_frames = np.random.rand(T, static_dim * len(windows))
        >>> variance_frames = np.random.rand(T, static_dim * len(windows))
        >>> static_features = G.mlpg(mean_frames, variance_frames, windows)
        >>> assert static_features.shape == (T, static_dim)

    See also:
        :func:`nnmnkwii.autograd.mlpg`

    """
    dtype = mean_frames.dtype
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
    # Perform dimension-wise generation
    y = np.zeros((T, static_dim), dtype=dtype)
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

    Parameters are same as :func:`nnmnkwii.paramgen.mlpg` except for
    ``grad_output``. See the function docmenent for what the parameters mean.

    Let :math:`d` is the index of static features, :math:`l` is the index
    of windows, gradients :math:`g_{d,l}` can be computed by:

    .. math::

        g_{d,l} = (\sum_{l} W_{l}^{T}P_{d,l}W_{l})^{-1} W_{l}^{T}P_{d,l}

    where :math:`W_{l}` is a banded window matrix and :math:`P_{d,l}` is a
    diagonal precision matrix.

    Assuming the variances are diagonals, MLPG can be performed in
    dimention-by-dimention efficiently.

    Let :math:`o_{d}` be ``T`` dimentional back-propagated gradients, the
    resulting gradients :math:`g'_{l,d}` to be propagated are
    computed as follows:

    .. math::

        g'_{d,l} = o_{d}^{T} g_{d,l}


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

            # Finally we get grad for a particular dimension
            grads[:, win_idx * static_dim + d] = grad_output[:, d].T.dot(grad)

    return grads


def full_window_mat(win_mats, T):
    """Given banded window matrices, compute cocatenated full window matrix.

    Args:
        win_mats (list): List of windows matrices given by :func:`build_win_mats`.
        T (int): Number of frames.

    Returns:
        numpy.ndarray: Cocatenated windows matrix (``T*num_windows x T``).
    """
    return _full_window_mat(win_mats, T)


def unit_variance_mlpg_matrix(windows, T):
    """Compute MLPG matrix assuming input is normalized to have unit-variances.

    Let :math:`\mu` is the input mean sequence (``num_windows*T x static_dim``),
    :math:`W` is a window matrix ``(T x num_windows*T)``, assuming input is
    normalized to have unit-variances, MLPG can be written as follows:

    .. math::

        y = R \mu

    where

    .. math::

        R = (W^{T} W)^{-1} W^{T}

    Here we call :math:`R` as the MLPG matrix.

    Args:
        windows: (list): List of windows.
        T (int): Number of frames.

    Returns:
        numpy.ndarray: MLPG matrix (``T x nun_windows*T``).

    See also:
        :func:`nnmnkwii.autograd.UnitVarianceMLPG`,
        :func:`nnmnkwii.paramgen.mlpg`.

    Examples:
        >>> from nnmnkwii import paramgen as G
        >>> import numpy as np
        >>> windows = [
        ...         (0, 0, np.array([1.0])),
        ...         (1, 1, np.array([-0.5, 0.0, 0.5])),
        ...         (1, 1, np.array([1.0, -2.0, 1.0])),
        ...     ]
        >>> G.unit_variance_mlpg_matrix(windows, 3)
        array([[  2.73835927e-01,   1.95121944e-01,   9.20177400e-02,
                  9.75609720e-02,  -9.09090936e-02,  -9.75609720e-02,
                 -3.52549881e-01,  -2.43902430e-02,   1.10864742e-02],
               [  1.95121944e-01,   3.41463417e-01,   1.95121944e-01,
                  1.70731708e-01,  -5.55111512e-17,  -1.70731708e-01,
                 -4.87804860e-02,  -2.92682916e-01,  -4.87804860e-02],
               [  9.20177400e-02,   1.95121944e-01,   2.73835927e-01,
                  9.75609720e-02,   9.09090936e-02,  -9.75609720e-02,
                  1.10864742e-02,  -2.43902430e-02,  -3.52549881e-01]], dtype=float32)
    """
    win_mats = build_win_mats(windows, T)
    sdw = np.max([win_mat.l + win_mat.u for win_mat in win_mats])

    P = bm.zeros(sdw, sdw, T)
    for win_index, win_mat in enumerate(win_mats):
        bm.dot_mm_plus_equals(win_mat.T, win_mat, target_bm=P)
    chol_bm = bla.cholesky(P, lower=True)
    Pinv = cholesky_inv_banded(chol_bm.full(), width=chol_bm.l + chol_bm.u + 1)

    cocatenated_window = full_window_mat(win_mats, T)
    return Pinv.dot(cocatenated_window.T).astype(np.float32)


def reshape_means(means, static_dim):
    """Reshape means (``T x D``) to (``T*num_windows x static_dim``).

    Args:
        means (numpy.ndarray): Means
        num_windows (int): Number of windows.

    Returns:
        numpy.ndarray: Reshaped means (``T*num_windows x static_dim``).
        No-op if already reshaped.

    Examples:
        >>> from nnmnkwii import paramgen as G
        >>> import numpy as np
        >>> T, static_dim = 2, 2
        >>> windows = [
        ...     (0, 0, np.array([1.0])),            # static
        ...     (1, 1, np.array([-0.5, 0.0, 0.5])), # delta
        ...     (1, 1, np.array([1.0, -2.0, 1.0])), # delta-delta
        ... ]
        >>> means = np.random.rand(T, static_dim * len(windows))
        >>> reshaped_means = G.reshape_means(means, static_dim)
        >>> assert reshaped_means.shape == (T*len(windows), static_dim)
    """
    T, D = means.shape
    if D == static_dim:
        # already reshaped case
        return means
    means = means.reshape(
        T, -1, static_dim).transpose(1, 0, 2).reshape(-1, static_dim)
    return means
