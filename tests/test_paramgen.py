from __future__ import division, print_function, absolute_import

import numpy as np

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


@attr("requires_bandmat")
def test_mlpg():
    from nnmnkwii import paramgen as G

    static_dim = 2
    T = 10

    windows_set = _get_windows_set()
    for windows in windows_set:
        means = np.random.rand(T, static_dim * len(windows))
        variances = np.tile(np.random.rand(static_dim * len(windows)), (T, 1))

        generated = G.mlpg(means, variances, windows)
        assert generated.shape == (T, static_dim)

    # Test variances correctly expanded
    for windows in windows_set:
        for dtype in [np.float32, np.float64]:
            means = np.random.rand(T, static_dim * len(windows)).astype(dtype)
            variances = np.random.rand(static_dim * len(windows)).astype(dtype)
            variances_frames = np.tile(variances, (T, 1))

            # Explicitly give variances over frame
            generated1 = G.mlpg(means, variances_frames, windows)
            # Give global variances. This will get expanded over frames
            # internally
            generated2 = G.mlpg(means, variances, windows)

            assert generated1.dtype == dtype
            assert np.allclose(generated1, generated2)


@attr("requires_bandmat")
def test_mlpg_window_full():
    from nnmnkwii import paramgen as G
    static_dim = 2
    T = 10

    def full_window_mat_native(win_mats, T):
        cocatenated_window = np.zeros((T * len(windows), T))
        for win_index, win_mat in enumerate(win_mats):
            win = win_mat.full()
            b = win_index * T
            cocatenated_window[b:b + T, :] = win
        return cocatenated_window

    for windows in _get_windows_set():
        win_mats = G.build_win_mats(windows, T)
        fullwin = G.full_window_mat(win_mats, T)
        assert fullwin.shape == (T * len(windows), T)
        assert np.allclose(full_window_mat_native(win_mats, T), fullwin)


@attr("requires_bandmat")
def test_unit_variance_mlpg():
    from nnmnkwii import paramgen as G
    static_dim = 2
    T = 10

    for windows in _get_windows_set():
        means = np.random.rand(T, static_dim * len(windows))
        variances = np.ones(static_dim * len(windows))
        y = G.mlpg(means, variances, windows)

        R = G.unit_variance_mlpg_matrix(windows, T)
        y_hat = R.dot(G.reshape_means(means, static_dim))
        assert np.allclose(y_hat, y)


@attr("requires_bandmat")
def test_reshape_means():
    from nnmnkwii import paramgen as G
    static_dim = 2
    T = 10

    for windows in _get_windows_set():
        means = np.random.rand(T, static_dim * len(windows))
        reshaped_means = G.reshape_means(means, static_dim)
        assert reshaped_means.shape == (T * len(windows), static_dim)
        reshaped_means2 = G.reshape_means(reshaped_means, static_dim)
        # Test if call on reshaped means doesn't change anything
        assert np.allclose(reshaped_means, reshaped_means2)
