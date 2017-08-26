from __future__ import with_statement, print_function, absolute_import

import numpy as np

# TODO: this may be removed in future.


def modspec(y, n=4096, norm=None):
    """Modulation spectrum computation

    Given an parameter trajectory (``T x D``), it computes modulation spectrum.
    Here we define modulation spectrum is power of discrete Fourier transform
    of parameter trajectory. See [1]_ for example application.

    .. [1] Takamichi, Shinnosuke, et al. "A postfilter to modify the modulation
      spectrum in HMM-based speech synthesis." Acoustics, Speech and Signal
      Processing (ICASSP), 2014 IEEE International Conference on. IEEE, 2014.

    Args:
        n : DFT length
        norm: Normalization mode. See :func:`numpy.fft.fft`

    Returns:
        numpy.ndarray: Modulation spectrum as ``T x n//2 + 1`` array.

    See also:
        :func:`nnmnkwii.autograd.modspec`

    Examples:
        >>> import numpy as np
        >>> from nnmnkwii import functions as F
        >>> generated = np.random.rand(10, 2)
        >>> ms = F.modspec(generated, n=16)
        >>> ms.shape
        (9, 2)
    """
    T, D = y.shape
    # DFT against time axis
    s_complex = np.fft.rfft(y, n=n, axis=0, norm=norm)
    assert s_complex.shape[0] == n // 2 + 1
    R, I = s_complex.real, s_complex.imag
    return R * R + I * I


def modphase(y, n=4096, norm=None):
    """Phase of modulation spectrum.

    Given an parameter trajectory, it computes phase of modulation spectrum.

    Args:
        n : DFT length
        norm: Normalization mode. See :func:`numpy.fft.fft`

    Returns:
        numpy.ndarray: Modulation spectrum as ``T x n//2 + 1`` complex array.
    """
    T, D = y.shape

    # DFT against time axis
    s_complex = np.fft.rfft(y, n=n, axis=0, norm=norm)
    return np.exp(1.j * np.angle(s_complex))
