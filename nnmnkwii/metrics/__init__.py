from __future__ import with_statement, print_function, absolute_import

import numpy as np
import math

_logdb_const = 10.0 / np.log(10.0) * np.sqrt(2.0)


# should work on torch and numpy arrays
def _sqrt(x):
    isnumpy = isinstance(x, np.ndarray)
    isscalar = np.isscalar(x)
    return np.sqrt(x) if isnumpy else math.sqrt(x) if isscalar else x.sqrt()


def _exp(x):
    isnumpy = isinstance(x, np.ndarray)
    isscalar = np.isscalar(x)
    return np.exp(x) if isnumpy else math.exp(x) if isscalar else x.exp()


def _sum(x):
    if isinstance(x, list) or isinstance(x, np.ndarray):
        return np.sum(x)
    return float(x.sum())


def melcd(X, Y, lengths=None):
    """Mel-cepstrum distortion (MCD).

    The function computes MCD for time-aligned mel-cepstrum sequences.

    Args:
        X (ndarray): Input mel-cepstrum, shape can be either of
          (``D``,), (``T x D``) or (``B x T x D``). Both Numpy and torch arrays
          are supported.
        Y (ndarray): Target mel-cepstrum, shape can be either of
          (``D``,), (``T x D``) or (``B x T x D``). Both Numpy and torch arrays
          are supported.
        lengths (list): Lengths of padded inputs. This should only be specified
          if you give mini-batch inputs.

    Returns:
        float: Mean mel-cepstrum distortion in dB.

    .. note::

        The function doesn't check if inputs are actually mel-cepstrum.
    """
    # summing against feature axis, and then take mean against time axis
    # Eq. (1a)
    # https://www.cs.cmu.edu/~awb/papers/sltu2008/kominek_black.sltu_2008.pdf
    if lengths is None:
        z = X - Y
        r = _sqrt((z * z).sum(-1))
        if not np.isscalar(r):
            r = r.mean()
        return _logdb_const * float(r)

    # Case for 1-dim features.
    if len(X.shape) == 2:
        # Add feature axis
        X, Y = X[:, :, None], Y[:, :, None]

    s = 0.0
    T = _sum(lengths)
    for x, y, length in zip(X, Y, lengths):
        x, y = x[:length], y[:length]
        z = x - y
        s += _sqrt((z * z).sum(-1)).sum()

    return _logdb_const * float(s) / float(T)


def mean_squared_error(X, Y, lengths=None):
    """Mean squared error (MSE).

    Args:
        X (ndarray): Input features, shape can be either of
          (``D``,), (``T x D``) or (``B x T x D``). Both Numpy and torch arrays
          are supported.
        Y (ndarray): Target features, shape can be either of
          (``D``,), (``T x D``) or (``B x T x D``). Both Numpy and torch arrays
          are supported.
        lengths (list): Lengths of padded inputs. This should only be specified
          if you give mini-batch inputs.

    Returns:
        float: Mean squared error.

    .. tip::

        The function supports 3D padded inputs, while
        :func:`sklearn.metrics.mean_squared_error` doesn't support.
    """
    if lengths is None:
        z = X - Y
        return math.sqrt(float((z * z).mean()))

    T = _sum(lengths) * X.shape[-1]
    s = 0.0
    for x, y, length in zip(X, Y, lengths):
        x, y = x[:length], y[:length]
        z = x - y
        s += (z * z).sum()

    return math.sqrt(float(s) / float(T))


def lf0_mean_squared_error(src_f0, src_vuv, tgt_f0, tgt_vuv,
                           lengths=None, linear_domain=False):
    """Mean squared error (MSE) for log-F0 sequences.

    MSE is computed for voiced segments.

    Args:
        src_f0 (ndarray): Input log-F0 sequences, shape can be either of
          (``T``,), (``B x T``) or (``B x T x 1``). Both Numpy and torch arrays
          are supported.
        src_vuv (ndarray): Input voiced/unvoiced flag array, shape can be either
          of (``T``, ), (``B x T``) or (``B x T x 1``).
        tgt_f0 (ndarray): Target log-F0 sequences, shape can be either of
          (``T``,), (``B x T``) or (``B x T x 1``). Both Numpy and torch arrays
          are supported.
        tgt_vuv (ndarray): Target voiced/unvoiced flag array, shape can be either
          of (``T``, ), (``B x T``) or (``B x T x 1``).
        lengths (list): Lengths of padded inputs. This should only be specified
          if you give mini-batch inputs.
        linear_domain (bool): Whether computes MSE on linear frequecy domain or
          log-frequency domain.

    Returns:
        float: mean squared error.
    """

    if lengths is None:
        voiced_indices = (src_vuv + tgt_vuv) >= 2
        x = src_f0[voiced_indices]
        y = tgt_f0[voiced_indices]
        if linear_domain:
            x, y = _exp(x), _exp(y)
        return mean_squared_error(x, y)

    T = 0
    s = 0.0
    for x, x_vuv, y, y_vuv, length in zip(
            src_f0, src_vuv, tgt_f0, tgt_vuv, lengths):
        x, x_vuv = x[:length], x_vuv[:length]
        y, y_vuv = y[:length], y_vuv[:length]
        voiced_indices = (x_vuv + y_vuv) >= 2
        T += voiced_indices.sum()
        x, y = x[voiced_indices], y[voiced_indices]
        if linear_domain:
            x, y = _exp(x), _exp(y)
        z = x - y
        s += (z * z).sum()

    return math.sqrt(float(s) / float(T))


def vuv_error(src_vuv, tgt_vuv, lengths=None):
    """Voice/unvoiced error rate computation

    Args:
        src_vuv (ndarray): Input voiced/unvoiced flag array shape can be either
          of (``T``, ), (``B x T``) or (``B x T x 1``).
        tgt_vuv (ndarray): Target voiced/unvoiced flag array shape can be either
          of (``T``, ), (``B x T``) or (``B x T x 1``).
        lengths (list): Lengths of padded inputs. This should only be specified
          if you give mini-batch inputs.

    Returns:
        float: voiced/unvoiced error rate (0 ~ 1).
    """
    if lengths is None:
        T = np.prod(src_vuv.shape)
        return float((src_vuv != tgt_vuv).sum()) / float(T)

    T = _sum(lengths)
    s = 0.0
    for x, y, length in zip(src_vuv, tgt_vuv, lengths):
        x, y = x[:length], y[:length]
        s += (x != y).sum()
    return float(s) / float(T)
