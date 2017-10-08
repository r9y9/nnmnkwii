from __future__ import division, print_function, absolute_import

from nnmnkwii.preprocessing.f0 import interp1d
from nnmnkwii.preprocessing import trim_zeros_frames, remove_zeros_frames
from nnmnkwii.preprocessing import adjast_frame_lengths, delta_features
from nnmnkwii.preprocessing import adjast_frame_length
from nnmnkwii.preprocessing.alignment import DTWAligner, IterativeDTWAligner
from nnmnkwii.preprocessing import modspec, modphase
from nnmnkwii.preprocessing import preemphasis, inv_preemphasis
from nnmnkwii import preprocessing as P
from nnmnkwii.util import example_file_data_sources_for_duration_model
from nnmnkwii.util import example_file_data_sources_for_acoustic_model
from nnmnkwii.util import example_audio_file
from nnmnkwii.datasets import FileSourceDataset, PaddedFileSourceDataset

from nose.tools import raises

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


def test_meanvar_incremental():
    np.random.seed(1234)
    N = 32
    X = np.random.randn(N, 100, 24)
    lengths = [len(x) for x in X]
    X_mean = np.mean(X, axis=(0, 1))
    X_var = np.var(X, axis=(0, 1))
    X_std = np.sqrt(X_var)

    # Check consistency with numpy
    X_mean_inc, X_var_inc = P.meanvar(X)
    assert np.allclose(X_mean, X_mean_inc)
    assert np.allclose(X_var, X_var_inc)

    # Split dataset and compute meanvar incrementaly
    X_a = X[:N // 2]
    X_b = X[N // 2:]
    X_mean_a, X_var_a, last_sample_count = P.meanvar(
        X_a, return_last_sample_count=True)
    assert last_sample_count == np.sum(lengths[:N // 2])
    X_mean_b, X_var_b = P.meanvar(
        X_b, mean_=X_mean_a, var_=X_var_a,
        last_sample_count=last_sample_count)
    assert np.allclose(X_mean, X_mean_b)
    assert np.allclose(X_var, X_var_b)

    # meanstd
    X_mean_a, X_std_a, last_sample_count = P.meanstd(
        X_a, return_last_sample_count=True)
    assert last_sample_count == np.sum(lengths[:N // 2])
    X_mean_b, X_std_b = P.meanstd(
        X_b, mean_=X_mean_a, var_=X_std_a**2,
        last_sample_count=last_sample_count)
    assert np.allclose(X_mean, X_mean_b)
    assert np.allclose(X_std, X_std_b)


def test_meanvar():
    # Pick acoustic features for testing
    _, X = example_file_data_sources_for_acoustic_model()
    X = FileSourceDataset(X)
    lengths = [len(x) for x in X]
    D = X[0].shape[-1]
    X_mean, X_var = P.meanvar(X)
    X_std = np.sqrt(X_var)
    assert np.isfinite(X_mean).all()
    assert np.isfinite(X_var).all()
    assert X_mean.shape[-1] == D
    assert X_var.shape[-1] == D

    _, X_std_hat = P.meanstd(X)
    assert np.allclose(X_std, X_std_hat)

    x = X[0]
    x_scaled = P.scale(x, X_mean, X_std)
    assert np.isfinite(x_scaled).all()

    # For padded dataset
    _, X = example_file_data_sources_for_acoustic_model()
    X = PaddedFileSourceDataset(X, 1000)
    # Should get same results with padded features
    X_mean_hat, X_var_hat = P.meanvar(X, lengths)
    assert np.allclose(X_mean, X_mean_hat)
    assert np.allclose(X_var, X_var_hat)

    # Inverse transform
    x = X[0]
    x_hat = P.inv_scale(P.scale(x, X_mean, X_std), X_mean, X_std)
    assert np.allclose(x, x_hat, atol=1e-7)


def test_minmax():
    # Pick linguistic features for testing
    X, _ = example_file_data_sources_for_acoustic_model()
    X = FileSourceDataset(X)
    lengths = [len(x) for x in X]
    D = X[0].shape[-1]
    X_min, X_max = P.minmax(X)
    assert np.isfinite(X_min).all()
    assert np.isfinite(X_max).all()

    x = X[0]
    x_scaled = P.minmax_scale(x, X_min, X_max, feature_range=(0, 0.99))
    assert np.max(x_scaled) <= 1
    assert np.min(x_scaled) >= 0
    assert np.isfinite(x_scaled).all()

    # Need to specify (min, max) or (scale_, min_)
    @raises(ValueError)
    def __test_raise1(x, X_min, X_max):
        P.minmax_scale(x)

    @raises(ValueError)
    def __test_raise2(x, X_min, X_max):
        P.inv_minmax_scale(x)

    __test_raise1(x, X_min, X_max)
    __test_raise2(x, X_min, X_max)

    # Explicit scale_ and min_
    min_, scale_ = P.minmax_scale_params(X_min, X_max, feature_range=(0, 0.99))
    x_scaled_hat = P.minmax_scale(x, min_=min_, scale_=scale_)
    assert np.allclose(x_scaled, x_scaled_hat)

    # For padded dataset
    X, _ = example_file_data_sources_for_acoustic_model()
    X = PaddedFileSourceDataset(X, 1000)
    # Should get same results with padded features
    X_min_hat, X_max_hat = P.minmax(X, lengths)
    assert np.allclose(X_min, X_min_hat)
    assert np.allclose(X_max, X_max_hat)

    # Inverse transform
    x = X[0]
    x_hat = P.inv_minmax_scale(P.minmax_scale(x, X_min, X_max), X_min, X_max)
    assert np.allclose(x, x_hat)

    x_hat = P.inv_minmax_scale(
        P.minmax_scale(x, scale_=scale_, min_=min_), scale_=scale_, min_=min_)
    assert np.allclose(x, x_hat)


def test_preemphasis():
    for dtype in [np.float32, np.float64]:
        np.random.seed(1234)
        x = np.random.rand(16000 * 5).astype(dtype)
        y = preemphasis(x, 0.97)
        assert x.shape == y.shape
        assert x.dtype == y.dtype
        x_hat = inv_preemphasis(y, 0.97)
        assert x_hat.dtype == x.dtype
        assert np.allclose(x_hat, x, atol=1e-5)


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


def test_adjast_frame_length_divisible():
    D = 5
    T = 10

    x = np.random.rand(T, D)
    assert T == adjast_frame_length(x, pad=True, divisible_by=1).shape[0]
    assert T == adjast_frame_length(x, pad=True, divisible_by=2).shape[0]
    print(adjast_frame_length(x, pad=True, divisible_by=3).shape[0])
    assert T + 2 == adjast_frame_length(x, pad=True, divisible_by=3).shape[0]
    assert T + 2 == adjast_frame_length(x, pad=True, divisible_by=4).shape[0]

    assert T == adjast_frame_length(x, pad=False, divisible_by=1).shape[0]
    assert T == adjast_frame_length(x, pad=False, divisible_by=2).shape[0]
    assert T - 1 == adjast_frame_length(x, pad=False, divisible_by=3).shape[0]
    assert T - 2 == adjast_frame_length(x, pad=False, divisible_by=4).shape[0]

    # Should preserve dtype
    for dtype in [np.float32, np.float64]:
        x = np.random.rand(T, D).astype(dtype)
        assert x.dtype == adjast_frame_length(x, pad=True, divisible_by=3).dtype
        assert x.dtype == adjast_frame_length(x, pad=False, divisible_by=3).dtype


def test_adjast_frame_lengths():
    D = 5
    T1 = 10
    T2 = 11

    x = np.random.rand(T1, D)
    y = np.random.rand(T2, D)
    x_hat, y_hat = adjast_frame_lengths(x, y, pad=True)
    assert x_hat.shape == y_hat.shape
    assert x_hat.shape[0] == 11

    x_hat, y_hat = adjast_frame_lengths(x, y, pad=False)
    assert x_hat.shape == y_hat.shape
    assert x_hat.shape[0] == 10

    x_hat, y_hat = adjast_frame_lengths(x, y, pad=True,
                                        divisible_by=2)
    assert x_hat.shape == y_hat.shape
    assert x_hat.shape[0] == 12

    x_hat, y_hat = adjast_frame_lengths(x, y, pad=False,
                                        divisible_by=2)
    assert x_hat.shape == y_hat.shape
    assert x_hat.shape[0] == 10

    # Divisible
    x_hat, y_hat = adjast_frame_lengths(x, y, pad=False,
                                        divisible_by=3)
    assert x_hat.shape == y_hat.shape
    assert x_hat.shape[0] == 9

    x_hat, y_hat = adjast_frame_lengths(x, y, pad=True,
                                        divisible_by=3)
    assert x_hat.shape == y_hat.shape
    assert x_hat.shape[0] == 12


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


def test_dtw_frame_length_adjastment():
    _, X = example_file_data_sources_for_duration_model()
    X = FileSourceDataset(X)
    X_unaligned = X.asarray()
    # This should trigger frame length adjastment
    Y_unaligned = np.pad(X_unaligned, [(0, 0), (5, 0), (0, 0)],
                         mode="constant", constant_values=0)
    Y_unaligned = Y_unaligned[:, :-5, :]
    for aligner in [DTWAligner(), IterativeDTWAligner(
            n_iter=1, max_iter_gmm=1, n_components_gmm=1)]:
        X_aligned, Y_aligned = aligner.transform((X_unaligned, Y_unaligned))
        assert X_aligned.shape == Y_aligned.shape


def test_dtw_aligner():
    x, fs = librosa.load(example_audio_file(), sr=None)
    assert fs == 16000
    x_fast = librosa.effects.time_stretch(x, 2.0)

    X = _get_mcep(x, fs)
    Y = _get_mcep(x_fast, fs)

    D = X.shape[-1]

    # Create padded pair
    X, Y = adjast_frame_lengths(X, Y, divisible_by=2)

    # Add utterance axis
    X = X.reshape(1, -1, D)
    Y = Y.reshape(1, -1, D)

    X_aligned, Y_aligned = DTWAligner().transform((X, Y))
    assert X_aligned.shape == Y_aligned.shape
    assert np.linalg.norm(X_aligned - Y_aligned) < np.linalg.norm(X - Y)

    X_aligned, Y_aligned = IterativeDTWAligner(
        n_iter=2, max_iter_gmm=10, n_components_gmm=2).transform((X, Y))
    assert X_aligned.shape == Y_aligned.shape
    assert np.linalg.norm(X_aligned - Y_aligned) < np.linalg.norm(X - Y)

    # Custom dist function
    from nnmnkwii.metrics import melcd
    X_aligned, Y_aligned = DTWAligner(dist=melcd).transform((X, Y))
    assert np.linalg.norm(X_aligned - Y_aligned) < np.linalg.norm(X - Y)


def test_modspec_reconstruct():
    static_dim = 2
    T = 64

    np.random.seed(1234)
    generated = np.random.rand(T, static_dim)

    for n in [64, 128]:
        ms, phase = P.modspec(generated, n=n, return_phase=True)  # ms = |X(w)|^2
        generated_hat = P.inv_modspec(ms, phase)[:T]
        assert np.allclose(generated, generated_hat)


def test_modspec_smoothing():
    static_dim = 2
    T = 64

    np.random.seed(1234)
    y = np.random.rand(T, static_dim)

    modfs = 200
    for log_domain in [True, False]:
        for norm in [None, "ortho"]:
            for n in [1024, 2048]:
                # Nyquist freq
                y_hat = P.modspec_smoothing(y, modfs, n=n, norm=norm,
                                            cutoff=modfs // 2,
                                            log_domain=log_domain)
                assert np.allclose(y, y_hat)

                # Smooth
                P.modspec_smoothing(y, modfs, n=n, norm=norm,
                                    cutoff=modfs // 4,
                                    log_domain=log_domain)

    # Cutoff frequency larger than modfs//2
    @raises(ValueError)
    def __test_invalid_param(y, modfs):
        P.modspec_smoothing(y, modfs, n=2048, cutoff=modfs // 2 + 1)

    # FFT size should larger than time length
    @raises(RuntimeError)
    def __test_invalid_time_length(y, modfs):
        P.modspec_smoothing(y, modfs, n=32, cutoff=modfs // 2)

    __test_invalid_time_length(y, modfs)
    __test_invalid_param(y, modfs)
