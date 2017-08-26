from __future__ import division, print_function, absolute_import

from nnmnkwii.preprocessing.f0 import interp1d
from nnmnkwii.preprocessing import trim_zeros_frames, remove_zeros_frames
from nnmnkwii.preprocessing import adjast_frame_length, delta_features
from nnmnkwii.util import example_audio_file
from nnmnkwii.preprocessing.alignment import DTWAligner, IterativeDTWAligner

from scipy.io import wavfile
import numpy as np
import pyworld
import librosa
import pysptk


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


def test_interp1d():
    f0 = np.random.rand(100, 1).astype(np.float32)
    f0[len(f0) // 2] = 0
    assert not np.all(f0 != 0)
    if0 = interp1d(f0)
    assert np.all(if0 != 0)


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


def test_delta_features():
    T = 5
    static_dim = 2
    x = np.random.rand(T, static_dim)
    for windows in _get_windows_set():
        y = delta_features(x, windows)
        assert y.shape == (T, static_dim * len(windows))


def _get_mcep(x, fs, frame_period=5, order=24):
    alpha = pysptk.util.mcepalpha(fs)
    x = x.astype(np.float64)
    f0, timeaxis = pyworld.dio(x, fs, frame_period=frame_period)
    f0 = pyworld.stonemask(x, f0, timeaxis, fs)
    spectrogram = pyworld.cheaptrick(x, f0, timeaxis, fs)
    spectrogram = trim_zeros_frames(spectrogram)
    mc = pysptk.sp2mc(spectrogram, order=order, alpha=alpha)
    return mc


def test_dtw_aligner():
    x, fs = librosa.load(example_audio_file(), sr=None)
    assert fs == 16000
    x_fast = librosa.effects.time_stretch(x, 2.0)

    X = _get_mcep(x, fs)
    Y = _get_mcep(x_fast, fs)

    D = X.shape[-1]

    # Create padded pair
    X, Y = adjast_frame_length(X, Y, ensure_even=True)

    # Add utterance axis
    X = X.reshape(1, -1, D)
    Y = Y.reshape(1, -1, D)

    X_aligned, Y_aligned = DTWAligner().transform((X, Y))
    assert X_aligned.shape == Y_aligned.shape
    assert np.linalg.norm(X_aligned - Y_aligned) < np.linalg.norm(X - Y)

    X_aligned, Y_aligned = IterativeDTWAligner(
        n_iter=3, max_iter_gmm=30, n_components_gmm=4).transform((X, Y))
    assert X_aligned.shape == Y_aligned.shape
    assert np.linalg.norm(X_aligned - Y_aligned) < np.linalg.norm(X - Y)
