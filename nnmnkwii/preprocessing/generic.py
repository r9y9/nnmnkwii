# coding: utf-8
from __future__ import division, print_function, absolute_import

import numpy as np
from sklearn.utils.extmath import _incremental_mean_and_var
from sklearn.preprocessing.data import _handle_zeros_in_scale
from scipy import signal


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


def trim_zeros_frames(x, eps=1e-7):
    """Remove trailling zeros frames.

    Similar to :func:`numpy.trim_zeros`, trimming trailing zeros features.

    Args:
        x (numpy.ndarray): Feature matrix, shape (``T x D``)
        eps (float): Values smaller than ``eps`` considered as zeros.

    Returns:
        numpy.ndarray: Trimmed 2d feature matrix, shape (``T' x D``)

    Examples:
        >>> import numpy as np
        >>> from nnmnkwii.preprocessing import trim_zeros_frames
        >>> x = np.random.rand(100,10)
        >>> y = trim_zeros_frames(x)
    """

    T, D = x.shape
    s = np.sum(np.abs(x), axis=1)
    s[s < eps] = 0.
    return x[:len(np.trim_zeros(s))]


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


def adjast_frame_length(x, pad=True, divisible_by=1):
    """Adjast frame length given a feature matrix.

    This adjast the number of frames of a given feature matrix to be divisible
    by ``divisible_by`` by padding zeros to the end or removing the last frames.

    Args:
        x (numpy.ndarray): Input 2d feature matrix, shape (``T x D``).
        pad (bool) : If True, pads zeros to the end, otherwise removes last few
          frames to ensure same frame lengths.
        divisible_by (int) : If ``divisible_by`` > 0, number of frames will be
          adjasted to be divisible by ``divisible_by``.

    Returns:
        numpy.ndarray: Adjasted feature matrix, of each shape (``T' x D``).

    Examples:
        >>> from nnmnkwii.preprocessing import adjast_frame_length
        >>> import numpy as np
        >>> x = np.zeros((10, 1))
        >>> x = adjast_frame_length(x, pad=True, divisible_by=3)
        >>> assert x.shape[0] == 12

    See also:
        :func:`nnmnkwii.preprocessing.adjast_frame_lengths`
    """
    Tx, D = x.shape

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
            x = np.vstack(
                (x, np.zeros((T - Tx, D), dtype=x.dtype)))
        else:
            x = x[:T]

    return x


def adjast_frame_lengths(x, y, pad=True, ensure_even=False, divisible_by=1):
    """Adjast frame lengths given two feature matrices.

    This ensures that two feature matrices have same number of frames, by
    padding zeros to the end or removing last frames.

    .. warning::

        ``ensure_even`` is deprecated and will be removed in v0.1.0.
        Use ``divisible_by=2`` instead.

    Args:
        x (numpy.ndarray): Input 2d feature matrix, shape (``T^1 x D``).
        y (numpy.ndarray): Input 2d feature matrix, shape (``T^2 x D``).
        pad (bool) : If True, pads zeros to the end, otherwise removes last few
          frames to ensure same frame lengths.
        ensure_even (bool) : If True, ensure number of frames to be even number.
        divisible_by (int) : If ``divisible_by`` > 0, number of frames will be
          adjasted to be divisible by ``divisible_by``.

    Returns:
        Tuple: Pair of adjasted feature matrices, of each shape (``T x D``).

    Examples:
        >>> from nnmnkwii.preprocessing import adjast_frame_lengths
        >>> import numpy as np
        >>> x = np.zeros((10, 1))
        >>> y = np.zeros((11, 1))
        >>> x, y = adjast_frame_lengths(x, y)
        >>> assert len(x) == len(y)

    See also:
        :func:`nnmnkwii.preprocessing.adjast_frame_length`
    """
    Tx, Dx = x.shape
    Ty, Dy = y.shape
    assert Dx == Dy

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
            x = np.vstack(
                (x, np.zeros((T - Tx, x.shape[-1]), dtype=x.dtype)))
        else:
            x = x[:T]

    if Ty != T:
        if Ty < T:
            y = np.vstack(
                (y, np.zeros((T - Ty, y.shape[-1]), dtype=y.dtype)))
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
