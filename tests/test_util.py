from __future__ import division, print_function, absolute_import

from nnmnkwii.util import trim_zeros_frames, remove_zeros_frames
from nnmnkwii.util import adjast_frame_length, apply_delta_windows
from nnmnkwii.util import example_audio_file
from scipy.io import wavfile
import numpy as np
import pyworld


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
        # Static + delta + delta-delta
        [
            (0, 0, np.array([1.0])),
            (1, 1, np.array([-0.5, 0.0, 0.5])),
            (1, 1, np.array([1.0, -2.0, 1.0])),
        ],
    ]

    T = 5
    static_dim = 2
    x = np.random.rand(T, static_dim)
    for windows in windows_set:
        y = apply_delta_windows(x, windows)
        assert y.shape == (T, static_dim * len(windows))
