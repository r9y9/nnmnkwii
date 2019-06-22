# coding: utf-8
from __future__ import division, print_function, absolute_import

import numpy as np
from sklearn.utils.extmath import _incremental_mean_and_var
from sklearn.preprocessing.data import _handle_zeros_in_scale
from scipy import signal


def _sign(x):
    isnumpy = isinstance(x, np.ndarray)
    isscalar = np.isscalar(x)
    return np.sign(x) if isnumpy or isscalar else x.sign()


def _log1p(x):
    isnumpy = isinstance(x, np.ndarray)
    isscalar = np.isscalar(x)
    return np.log1p(x) if isnumpy or isscalar else x.log1p()


def _abs(x):
    isnumpy = isinstance(x, np.ndarray)
    isscalar = np.isscalar(x)
    return np.abs(x) if isnumpy or isscalar else x.abs()


def _asint(x):
    # ugly wrapper to support torch/numpy arrays
    isnumpy = isinstance(x, np.ndarray)
    isscalar = np.isscalar(x)
    return x.astype(np.int) if isnumpy else int(x) if isscalar else x.long()


def _asfloat(x):
    # ugly wrapper to support torch/numpy arrays
    isnumpy = isinstance(x, np.ndarray)
    isscalar = np.isscalar(x)
    return x.astype(np.float32) if isnumpy else float(x) if isscalar else x.float()


def mulaw(x, mu=256):
    """Mu-Law companding

    Method described in paper [1]_.

    .. math::

        f(x) = sign(x) \ln (1 + \mu |x|) / \ln (1 + \mu)

    Args:
        x (array-like): Input signal. Each value of input signal must be in
          range of [-1, 1].
        mu (number): Compression parameter ``μ``.

    Returns:
        array-like: Compressed signal ([-1, 1])

    See also:
        :func:`nnmnkwii.preprocessing.inv_mulaw`
        :func:`nnmnkwii.preprocessing.mulaw_quantize`
        :func:`nnmnkwii.preprocessing.inv_mulaw_quantize`

    .. [1] Brokish, Charles W., and Michele Lewis. "A-law and mu-law companding
        implementations using the tms320c54x." SPRA163 (1997).
    """
    return _sign(x) * _log1p(mu * _abs(x)) / _log1p(mu)


def inv_mulaw(y, mu=256):
    """Inverse of mu-law companding (mu-law expansion)

    .. math::

        f^{-1}(x) = sign(y) (1 / \mu) (1 + \mu)^{|y|} - 1)

    Args:
        y (array-like): Compressed signal. Each value of input signal must be in
          range of [-1, 1].
        mu (number): Compression parameter ``μ``.

    Returns:
        array-like: Uncomprresed signal (-1 <= x <= 1)

    See also:
        :func:`nnmnkwii.preprocessing.inv_mulaw`
        :func:`nnmnkwii.preprocessing.mulaw_quantize`
        :func:`nnmnkwii.preprocessing.inv_mulaw_quantize`
    """
    return _sign(y) * (1.0 / mu) * ((1.0 + mu)**_abs(y) - 1.0)


def mulaw_quantize(x, mu=256):
    """Mu-Law companding + quantize

    Args:
        x (array-like): Input signal. Each value of input signal must be in
          range of [-1, 1].
        mu (number): Compression parameter ``μ``.

    Returns:
        array-like: Quantized signal (dtype=int)
          - y ∈ [0, mu] if x ∈ [-1, 1]
          - y ∈ [0, mu) if x ∈ [-1, 1)

    .. note::
        If you want to get quantized values of range [0, mu) (not [0, mu]),
        then you need to provide input signal of range [-1, 1).

    Examples:
        >>> from scipy.io import wavfile
        >>> import pysptk
        >>> import numpy as np
        >>> from nnmnkwii import preprocessing as P
        >>> fs, x = wavfile.read(pysptk.util.example_audio_file())
        >>> x = (x / 32768.0).astype(np.float32)
        >>> y = P.mulaw_quantize(x)
        >>> print(y.min(), y.max(), y.dtype)
        15 246 int64

    See also:
        :func:`nnmnkwii.preprocessing.mulaw`
        :func:`nnmnkwii.preprocessing.inv_mulaw`
        :func:`nnmnkwii.preprocessing.inv_mulaw_quantize`
    """
    y = mulaw(x, mu)
    # scale [-1, 1] to [0, mu]
    return _asint((y + 1) / 2 * mu)


def inv_mulaw_quantize(y, mu=256):
    """Inverse of mu-law companding + quantize

    Args:
        y (array-like): Quantized signal (∈ [0, mu]).
        mu (number): Compression parameter ``μ``.

    Returns:
        array-like: Uncompressed signal ([-1, 1])

    Examples:
        >>> from scipy.io import wavfile
        >>> import pysptk
        >>> import numpy as np
        >>> from nnmnkwii import preprocessing as P
        >>> fs, x = wavfile.read(pysptk.util.example_audio_file())
        >>> x = (x / 32768.0).astype(np.float32)
        >>> x_hat = P.inv_mulaw_quantize(P.mulaw_quantize(x))
        >>> x_hat = (x_hat * 32768).astype(np.int16)

    See also:
        :func:`nnmnkwii.preprocessing.mulaw`
        :func:`nnmnkwii.preprocessing.inv_mulaw`
        :func:`nnmnkwii.preprocessing.mulaw_quantize`
    """
    # [0, m) to [-1, 1]
    y = 2 * _asfloat(y) / mu - 1
    return inv_mulaw(y, mu)


def preemphasis(x, coef=0.97):
    """Pre-emphasis

    Args:
        x (1d-array): Input signal.
        coef (float): Pre-emphasis coefficient.

    Returns:
        array: Output filtered signal.

    See also:
        :func:`inv_preemphasis`

    Examples:
        >>> from nnmnkwii.util import example_audio_file
        >>> from scipy.io import wavfile
        >>> fs, x = wavfile.read(example_audio_file())
        >>> x = x.astype(np.float64)
        >>> from nnmnkwii import preprocessing as P
        >>> y = P.preemphasis(x, coef=0.97)
        >>> assert x.shape == y.shape
    """
    b = np.array([1., -coef], x.dtype)
    a = np.array([1.], x.dtype)
    return signal.lfilter(b, a, x)


def inv_preemphasis(x, coef=0.97):
    """Inverse operation of pre-emphasis

    Args:
        x (1d-array): Input signal.
        coef (float): Pre-emphasis coefficient.

    Returns:
        array: Output filtered signal.

    See also:
        :func:`preemphasis`

    Examples:
        >>> from nnmnkwii.util import example_audio_file
        >>> from scipy.io import wavfile
        >>> fs, x = wavfile.read(example_audio_file())
        >>> x = x.astype(np.float64)
        >>> from nnmnkwii import preprocessing as P
        >>> x_hat = P.inv_preemphasis(P.preemphasis(x, coef=0.97), coef=0.97)
        >>> assert np.allclose(x, x_hat)
    """
    b = np.array([1.], x.dtype)
    a = np.array([1., -coef], x.dtype)
    return signal.lfilter(b, a, x)


def _delta(x, window):
    return np.correlate(x, window, mode="same")


def _apply_delta_window(x, window):
    """Returns delta features given a static features and a window.

    Args:
        x (numpy.ndarray): Input static features, of shape (``T x D``).
        window (numpy.ndarray): Window coefficients.

    Returns:
        (ndarray): Delta features, shape　(``T x D``).
    """
    T, D = x.shape
    y = np.zeros_like(x)
    for d in range(D):
        y[:, d] = _delta(x[:, d], window)
    return y


def delta_features(x, windows):
    """Compute delta features and combine them.

    This function computes delta features given delta windows, and then
    returns combined features (e.g., static + delta + delta-delta).
    Note that if you want to keep static features, you need to give
    static window as well as delta windows.

    Args:
        x (numpy.ndarray): Input static features, of shape (``T x D``).
        y (list): List of windows. See :func:`nnmnkwii.paramgen.mlpg` for what
            the delta window means.

    Returns:
        numpy.ndarray: static + delta features (``T x (D * len(windows)``).

    Examples:
        >>> from nnmnkwii.preprocessing import delta_features
        >>> windows = [
        ...         (0, 0, np.array([1.0])),            # static
        ...         (1, 1, np.array([-0.5, 0.0, 0.5])), # delta
        ...         (1, 1, np.array([1.0, -2.0, 1.0])), # delta-delta
        ...     ]
        >>> T, static_dim = 10, 24
        >>> x = np.random.rand(T, static_dim)
        >>> y = delta_features(x, windows)
        >>> assert y.shape == (T, static_dim * len(windows))
    """
    T, D = x.shape
    assert len(windows) > 0
    combined_features = np.empty((T, D * len(windows)), dtype=x.dtype)
    for idx, (_, _, window) in enumerate(windows):
        combined_features[:, D * idx:D * idx +
                          D] = _apply_delta_window(x, window)
    return combined_features


def trim_zeros_frames(x, eps=1e-7, trim='b'):
    """Remove leading and/or trailing zeros frames.

    Similar to :func:`numpy.trim_zeros`, trimming trailing zeros features.

    Args:
        x (numpy.ndarray): Feature matrix, shape (``T x D``)
        eps (float): Values smaller than ``eps`` considered as zeros.
        trim (string): Representing trim from where.

    Returns:
        numpy.ndarray: Trimmed 2d feature matrix, shape (``T' x D``)

    Examples:
        >>> import numpy as np
        >>> from nnmnkwii.preprocessing import trim_zeros_frames
        >>> x = np.random.rand(100,10)
        >>> y = trim_zeros_frames(x)
    """

    assert trim in {'f', 'b', 'fb'}

    T, D = x.shape
    s = np.sum(np.abs(x), axis=1)
    s[s < eps] = 0.

    if trim == 'f':
        return x[len(x) - len(np.trim_zeros(s, trim=trim)):]
    elif trim == 'b':
        end = len(np.trim_zeros(s, trim=trim)) - len(x)
        if end == 0:
            return x
        else:
            return x[: end]
    elif trim == 'fb':
        f = len(np.trim_zeros(s, trim='f'))
        b = len(np.trim_zeros(s, trim='b'))
        end = b - len(x)
        if end == 0:
            return x[len(x) - f:]
        else:
            return x[len(x) - f: end]


def remove_zeros_frames(x, eps=1e-7):
    """Remove zeros frames.

    Given a feature matrix, remove all zeros frames as well as trailing ones.

    Args:
        x (numpy.ndarray): 2d feature matrix, shape (``T x D``)
        eps (float): Values smaller than ``eps`` considered as zeros.

    Returns:
        numpy.ndarray: Zeros-removed 2d feature matrix, shape (``T' x D``).

    Examples:
        >>> import numpy as np
        >>> from nnmnkwii.preprocessing import remove_zeros_frames
        >>> x = np.random.rand(100,10)
        >>> y = remove_zeros_frames(x)
    """
    T, D = x.shape
    s = np.sum(np.abs(x), axis=1)
    s[s < eps] = 0.
    return x[s > eps]


def adjust_frame_length(x, pad=True, divisible_by=1, **kwargs):
    """Adjust frame length given a feature vector or matrix.

    This adjust the number of frames of a given feature vector or matrix to be
    divisible by ``divisible_by`` by padding to the end or removing the last
    few frames. Default uses zero-padding.

    Args:
        x (numpy.ndarray): Input 1d or 2d array, shape (``T,`` or ``T x D``).
        pad (bool) : If True, pads values to the end, otherwise removes last few
          frames to ensure same frame lengths.
        divisible_by (int) : If ``divisible_by`` > 0, number of frames will be
          adjusted to be divisible by ``divisible_by``.
        kwargs (dict): Keyword argments passed to :func:`numpy.pad`. Default is
          mode = ``constant``, which means zero padding.

    Returns:
        numpy.ndarray: adjusted array, of each shape (``T`` or ``T' x D``).

    Examples:
        >>> from nnmnkwii.preprocessing import adjust_frame_length
        >>> import numpy as np
        >>> x = np.zeros((10, 1))
        >>> x = adjust_frame_length(x, pad=True, divisible_by=3)
        >>> assert x.shape[0] == 12

    See also:
        :func:`nnmnkwii.preprocessing.adjust_frame_lengths`
    """
    kwargs.setdefault("mode", "constant")

    assert x.ndim == 2 or x.ndim == 1
    Tx = x.shape[0]

    if divisible_by > 1:
        rem = Tx % divisible_by
        if rem == 0:
            T = Tx
        else:
            if pad:
                T = Tx + divisible_by - rem
            else:
                T = Tx - rem
    else:
        T = Tx

    if Tx != T:
        if T > Tx:
            if x.ndim == 1:
                x = np.pad(x, (0, T - Tx), **kwargs)
            elif x.ndim == 2:
                x = np.pad(x, [(0, T - Tx), (0, 0)], **kwargs)
        else:
            x = x[:T]

    return x


def adjust_frame_lengths(x, y, pad=True, ensure_even=False, divisible_by=1,
                         **kwargs):
    """Adjust frame lengths given two feature vectors or matrices.

    This ensures that two feature vectors or matrices have same number of
    frames, by padding to the end or removing the last few frames.
    Default uses zero-padding.

    .. warning::

        ``ensure_even`` is deprecated and will be removed in v0.1.0.
        Use ``divisible_by=2`` instead.

    Args:
        x (numpy.ndarray): Input 2d feature matrix, shape (``T^1 x D``).
        y (numpy.ndarray): Input 2d feature matrix, shape (``T^2 x D``).
        pad (bool) : If True, pads values to the end, otherwise removes last few
          frames to ensure same frame lengths.
        divisible_by (int) : If ``divisible_by`` > 0, number of frames will be
          adjusted to be divisible by ``divisible_by``.
        kwargs (dict): Keyword argments passed to :func:`numpy.pad`. Default is
          mode = ``constant``, which means zero padding.

    Returns:
        Tuple: Pair of adjusted feature matrices, of each shape (``T x D``).

    Examples:
        >>> from nnmnkwii.preprocessing import adjust_frame_lengths
        >>> import numpy as np
        >>> x = np.zeros((10, 1))
        >>> y = np.zeros((11, 1))
        >>> x, y = adjust_frame_lengths(x, y)
        >>> assert len(x) == len(y)

    See also:
        :func:`nnmnkwii.preprocessing.adjust_frame_length`
    """
    assert x.ndim in [1, 2] and y.ndim in [1, 2]
    kwargs.setdefault("mode", "constant")
    Tx = x.shape[0]
    Ty = y.shape[0]
    if x.ndim == 2:
        assert x.shape[-1] == y.shape[-1]

    if ensure_even:
        divisible_by = 2

    if pad:
        T = max(Tx, Ty)
        if divisible_by > 1:
            rem = T % divisible_by
            if rem != 0:
                T = T + divisible_by - rem
    else:
        T = min(Tx, Ty)
        if divisible_by > 1:
            rem = T % divisible_by
            T = T - rem

    if Tx != T:
        if Tx < T:
            if x.ndim == 1:
                x = np.pad(x, (0, T - Tx), **kwargs)
            elif x.ndim == 2:
                x = np.pad(x, [(0, T - Tx), (0, 0)], **kwargs)
        else:
            x = x[:T]

    if Ty != T:
        if Ty < T:
            if y.ndim == 1:
                y = np.pad(y, (0, T - Ty), **kwargs)
            elif y.ndim == 2:
                y = np.pad(y, [(0, T - Ty), (0, 0)], **kwargs)
        else:
            y = y[:T]

    return x, y


def meanvar(dataset, lengths=None, mean_=0., var_=0.,
            last_sample_count=0, return_last_sample_count=False):
    """Mean/variance computation given a iterable dataset

    Dataset can have variable length samples. In that cases, you need to
    explicitly specify lengths for all the samples.

    Args:
        dataset (nnmnkwii.datasets.Dataset): Dataset
        lengths: (list): Frame lengths for each dataset sample.
        mean\_ (array or scalar): Initial value for mean vector.
        var\_ (array or scaler): Initial value for variance vector.
        last_sample_count (int): Last sample count. Default is 0. If you set
          non-default ``mean_`` and ``var_``, you need to set
          ``last_sample_count`` property. Typically this will be the number of
          time frames ever seen.
        return_last_sample_count (bool): Return ``last_sample_count`` if True.

    Returns:
        tuple: Mean and variance for each dimention. If
          ``return_last_sample_count`` is True, returns ``last_sample_count``
          as well.

    See also:
        :func:`nnmnkwii.preprocessing.meanstd`, :func:`nnmnkwii.preprocessing.scale`

    Examples:
        >>> from nnmnkwii.preprocessing import meanvar
        >>> from nnmnkwii.util import example_file_data_sources_for_acoustic_model
        >>> from nnmnkwii.datasets import FileSourceDataset
        >>> X, Y = example_file_data_sources_for_acoustic_model()
        >>> X, Y = FileSourceDataset(X), FileSourceDataset(Y)
        >>> lengths = [len(y) for y in Y]
        >>> data_mean, data_var = meanvar(Y, lengths)
    """
    dtype = dataset[0].dtype

    for idx, x in enumerate(dataset):
        if lengths is not None:
            x = x[:lengths[idx]]
        mean_, var_, _ = _incremental_mean_and_var(
            x, mean_, var_, last_sample_count)
        last_sample_count += len(x)
    mean_, var_ = mean_.astype(dtype), var_.astype(dtype)

    if return_last_sample_count:
        return mean_, var_, last_sample_count
    else:
        return mean_, var_


def meanstd(dataset, lengths=None, mean_=0., var_=0.,
            last_sample_count=0, return_last_sample_count=False):
    """Mean/std-deviation computation given a iterable dataset

    Dataset can have variable length samples. In that cases, you need to
    explicitly specify lengths for all the samples.

    Args:
        dataset (nnmnkwii.datasets.Dataset): Dataset
        lengths: (list): Frame lengths for each dataset sample.
        mean\_ (array or scalar): Initial value for mean vector.
        var\_ (array or scaler): Initial value for variance vector.
        last_sample_count (int): Last sample count. Default is 0. If you set
          non-default ``mean_`` and ``var_``, you need to set
          ``last_sample_count`` property. Typically this will be the number of
          time frames ever seen.
        return_last_sample_count (bool): Return ``last_sample_count`` if True.

    Returns:
        tuple: Mean and variance for each dimention. If
          ``return_last_sample_count`` is True, returns ``last_sample_count``
          as well.

    See also:
        :func:`nnmnkwii.preprocessing.meanvar`, :func:`nnmnkwii.preprocessing.scale`

    Examples:
        >>> from nnmnkwii.preprocessing import meanstd
        >>> from nnmnkwii.util import example_file_data_sources_for_acoustic_model
        >>> from nnmnkwii.datasets import FileSourceDataset
        >>> X, Y = example_file_data_sources_for_acoustic_model()
        >>> X, Y = FileSourceDataset(X), FileSourceDataset(Y)
        >>> lengths = [len(y) for y in Y]
        >>> data_mean, data_std = meanstd(Y, lengths)
    """
    ret = meanvar(dataset, lengths, mean_, var_,
                  last_sample_count, return_last_sample_count)
    m, v = ret[0], ret[1]
    v = _handle_zeros_in_scale(np.sqrt(v))
    if return_last_sample_count:
        assert len(ret) == 3
        return m, v, ret[2]
    else:
        return m, v


def minmax(dataset, lengths=None):
    """Min/max computation given a iterable dataset

    Dataset can have variable length samples. In that cases, you need to
    explicitly specify lengths for all the samples.

    Args:
        dataset (nnmnkwii.datasets.Dataset): Dataset
        lengths: (list): Frame lengths for each dataset sample.

    See also:
        :func:`nnmnkwii.preprocessing.minmax_scale`

    Examples:
        >>> from nnmnkwii.preprocessing import minmax
        >>> from nnmnkwii.util import example_file_data_sources_for_acoustic_model
        >>> from nnmnkwii.datasets import FileSourceDataset
        >>> X, Y = example_file_data_sources_for_acoustic_model()
        >>> X, Y = FileSourceDataset(X), FileSourceDataset(Y)
        >>> lengths = [len(x) for x in X]
        >>> data_min, data_max = minmax(X, lengths)
    """
    max_ = -np.inf
    min_ = np.inf

    for idx, x in enumerate(dataset):
        if lengths is not None:
            x = x[:lengths[idx]]
        min_ = np.minimum(min_, np.min(x, axis=(0,)))
        max_ = np.maximum(max_, np.max(x, axis=(0,)))

    return min_, max_


def scale(x, data_mean, data_std):
    """Mean/variance scaling.

    Given mean and variances, apply mean-variance normalization to data.

    Args:
        x (array): Input data
        data_mean (array): Means for each feature dimention.
        data_std (array): Standard deviation for each feature dimention.

    Returns:
        array: Scaled data.

    Examples:
        >>> from nnmnkwii.preprocessing import meanstd, scale
        >>> from nnmnkwii.util import example_file_data_sources_for_acoustic_model
        >>> from nnmnkwii.datasets import FileSourceDataset
        >>> X, Y = example_file_data_sources_for_acoustic_model()
        >>> X, Y = FileSourceDataset(X), FileSourceDataset(Y)
        >>> lengths = [len(y) for y in Y]
        >>> data_mean, data_std = meanstd(Y, lengths)
        >>> scaled_y = scale(Y[0], data_mean, data_std)

    See also:
        :func:`nnmnkwii.preprocessing.inv_scale`
    """
    return (x - data_mean) / _handle_zeros_in_scale(data_std, copy=False)


def inv_scale(x, data_mean, data_std):
    """Inverse tranform of mean/variance scaling.

    Given mean and variances, apply mean-variance denormalization to data.

    Args:
        x (array): Input data
        data_mean (array): Means for each feature dimention.
        data_std (array): Standard deviation for each feature dimention.

    Returns:
        array: Denormalized data.

    See also:
        :func:`nnmnkwii.preprocessing.scale`
    """
    return data_std * x + data_mean


def __minmax_scale_factor(data_min, data_max, feature_range):
    data_range = data_max - data_min
    scale = (feature_range[1] - feature_range[0]) / \
        _handle_zeros_in_scale(data_range, copy=False)
    return scale


def minmax_scale_params(data_min, data_max, feature_range=(0, 1)):
    """Compute parameters required to perform min/max scaling.

    Given data min, max and feature range, computes scalining factor and
    minimum value. Min/max scaling can be done as follows:

    .. code-block:: python

        x_scaled = x * scale_ + min_

    Args:
        x (array): Input data
        data_min (array): Data min for each feature dimention.
        data_max (array): Data max for each feature dimention.
        feature_range (array like): Feature range.

    Returns:
        tuple: Minimum value and scaling factor for scaled data.

    Examples:
        >>> from nnmnkwii.preprocessing import minmax, minmax_scale
        >>> from nnmnkwii.preprocessing import minmax_scale_params
        >>> from nnmnkwii.util import example_file_data_sources_for_acoustic_model
        >>> from nnmnkwii.datasets import FileSourceDataset
        >>> X, Y = example_file_data_sources_for_acoustic_model()
        >>> X, Y = FileSourceDataset(X), FileSourceDataset(Y)
        >>> data_min, data_max = minmax(X)
        >>> min_, scale_ = minmax_scale_params(data_min, data_max)
        >>> scaled_x = minmax_scale(X[0], min_=min_, scale_=scale_)

    See also:
        :func:`nnmnkwii.preprocessing.minmax_scale`,
        :func:`nnmnkwii.preprocessing.inv_minmax_scale`
    """
    scale_ = __minmax_scale_factor(data_min, data_max, feature_range)
    min_ = feature_range[0] - data_min * scale_
    return min_, scale_


def minmax_scale(x, data_min=None, data_max=None, feature_range=(0, 1),
                 scale_=None, min_=None):
    """Min/max scaling for given a single data.

    Given data min, max and feature range, apply min/max normalization to data.
    Optionally, you can get a little performance improvement to give scaling
    factor (``scale_``) and minimum value (``min_``) used in scaling explicitly.
    Those values can be computed by
    :func:`nnmnkwii.preprocessing.minmax_scale_params`.

    .. note::

        If ``scale_`` and ``min_`` are given, ``feature_range`` will be ignored.

    Args:
        x (array): Input data
        data_min (array): Data min for each feature dimention.
        data_max (array): Data max for each feature dimention.
        feature_range (array like): Feature range.
        scale\_ ([optional]array): Scaling factor.
        min\_ ([optional]array): Minimum value for scaling.

    Returns:
        array: Scaled data.

    Raises:
        ValueError: If (``data_min``, ``data_max``) or
          (``scale_`` and ``min_``) are not specified.

    See also:
        :func:`nnmnkwii.preprocessing.inv_minmax_scale`,
        :func:`nnmnkwii.preprocessing.minmax_scale_params`

    Examples:
        >>> from nnmnkwii.preprocessing import minmax, minmax_scale
        >>> from nnmnkwii.util import example_file_data_sources_for_acoustic_model
        >>> from nnmnkwii.datasets import FileSourceDataset
        >>> X, Y = example_file_data_sources_for_acoustic_model()
        >>> X, Y = FileSourceDataset(X), FileSourceDataset(Y)
        >>> data_min, data_max = minmax(X)
        >>> scaled_x = minmax_scale(X[0], data_min, data_max)
    """
    if (scale_ is None or min_ is None) and (data_min is None or data_max is None):
        raise ValueError("""
`data_min` and `data_max` or `scale_` and `min_` must be specified to perform minmax scale""")
    if scale_ is None:
        scale_ = __minmax_scale_factor(data_min, data_max, feature_range)
    if min_ is None:
        min_ = feature_range[0] - data_min * scale_
    return x * scale_ + min_


def inv_minmax_scale(x, data_min=None, data_max=None, feature_range=(0, 1),
                     scale_=None, min_=None):
    """Inverse transform of min/max scaling for given a single data.

    Given data min, max and feature range, apply min/max denormalization to data.

    .. note::

        If ``scale_`` and ``min_`` are given, ``feature_range`` will be ignored.

    Args:
        x (array): Input data
        data_min (array): Data min for each feature dimention.
        data_max (array): Data max for each feature dimention.
        feature_range (array like): Feature range.
        scale\_ ([optional]array): Scaling factor.
        min\_ ([optional]array): Minimum value for scaling.

    Returns:
        array: Scaled data.

    Raises:
        ValueError: If (``data_min``, ``data_max``) or
          (``scale_`` and ``min_``) are not specified.

    See also:
        :func:`nnmnkwii.preprocessing.minmax_scale`,
        :func:`nnmnkwii.preprocessing.minmax_scale_params`
    """
    if (scale_ is None or min_ is None) and (data_min is None or data_max is None):
        raise ValueError("""
`data_min` and `data_max` or `scale_` and `min_` must be specified to perform inverse of minmax scale""")
    if scale_ is None:
        scale_ = __minmax_scale_factor(data_min, data_max, feature_range)
    if min_ is None:
        min_ = feature_range[0] - data_min * scale_
    return (x - min_) / scale_
