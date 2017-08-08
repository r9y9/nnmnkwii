from __future__ import division, print_function, absolute_import

from nnmnkwii import functions as F
import numpy as np


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


def test_mlpg():
    static_dim = 2
    T = 10

    for windows in _get_windows_set():
        means = np.random.rand(T, static_dim * len(windows))
        variances = np.tile(np.random.rand(static_dim * len(windows)), (T, 1))

        generated = F.mlpg(means, variances, windows)
        assert generated.shape == (T, static_dim)


def test_modspec_reconstruct():
    static_dim = 2
    T = 64

    np.random.seed(1234)
    generated = np.random.rand(T, static_dim)

    for n in [64, 128]:
        ms = F.modspec(generated, n=n)  # ms = |X(w)|^2
        ms_phase = F.modphase(generated, n=n)
        complex_ms = np.sqrt(ms) * ms_phase  # = |X(w)| * phase
        generated_hat = np.fft.irfft(complex_ms, n=n, axis=0)[:T]
        assert np.allclose(generated, generated_hat)
