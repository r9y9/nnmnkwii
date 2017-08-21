from __future__ import division, print_function, absolute_import

from nnmnkwii.util import trim_zeros_frames, remove_zeros_frames
from nnmnkwii.util import adjast_frame_length, apply_delta_windows
from nnmnkwii.util import example_audio_file
from nnmnkwii import functions as F
from nnmnkwii.util.linalg import cholesky_inv, cholesky_inv_banded

from scipy.io import wavfile
import numpy as np
import pyworld
import scipy.linalg
import bandmat as bm

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


def test_trim_remove_zeros_frames():
    fs, x = wavfile.read(example_audio_file())
    frame_period = 5

    x = x.astype(np.float64)
    f0, timeaxis = pyworld.dio(x, fs, frame_period=frame_period)
    spectrogram = pyworld.cheaptrick(x, f0, timeaxis, fs)
    aperiodicity = pyworld.d4c(x, f0, timeaxis, fs)

    for mat in [spectrogram, aperiodicity]:
        trimmed = trim_zeros_frames(mat)
        assert trimmed.shape[1] == mat.shape[1]

    for mat in [spectrogram, aperiodicity]:
        trimmed = remove_zeros_frames(mat)
        assert trimmed.shape[1] == mat.shape[1]


def test_adjast_frame_length():
    D = 5
    T1 = 10
    T2 = 11

    x = np.random.rand(T1, D)
    y = np.random.rand(T2, D)
    x_hat, y_hat = adjast_frame_length(x, y, pad=True)
    assert x_hat.shape == y_hat.shape
    assert x_hat.shape[0] == 11

    x_hat, y_hat = adjast_frame_length(x, y, pad=False)
    assert x_hat.shape == y_hat.shape
    assert x_hat.shape[0] == 10

    x_hat, y_hat = adjast_frame_length(x, y, pad=True,
                                       ensure_even=True)
    assert x_hat.shape == y_hat.shape
    assert x_hat.shape[0] == 12

    x_hat, y_hat = adjast_frame_length(x, y, pad=False,
                                       ensure_even=True)
    assert x_hat.shape == y_hat.shape
    assert x_hat.shape[0] == 10


def test_apply_delta_windows():
    T = 5
    static_dim = 2
    x = np.random.rand(T, static_dim)
    for windows in _get_windows_set():
        y = apply_delta_windows(x, windows)
        assert y.shape == (T, static_dim * len(windows))

def _get_banded_test_mat(win_mats, T):
    sdw = max([win_mat.l + win_mat.u for win_mat in win_mats])
    P = bm.zeros(sdw, sdw, T)
    for win_index, win_mat in enumerate(win_mats):
        bm.dot_mm_plus_equals(win_mat.T, win_mat, target_bm=P)
    return P

def test_linalg_choleskey_inv():
    for windows in _get_windows_set():
        for T in [5, 10]:
            win_mats = F.build_win_mats(windows, T)
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
