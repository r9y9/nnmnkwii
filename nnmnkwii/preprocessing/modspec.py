from __future__ import with_statement, print_function, absolute_import

import numpy as np

# TODO: this may be removed in future.


def modspec(x, n=4096, norm=None, return_phase=False):
    """Modulation spectrum (MS) computation

    Given a parameter trajectory, it computes modulation spectrum. In the
    library, we define modulation spectrum as power of discrete Fourier transform
    of parameter trajectory across time-axis. See [1]_ for example application.

    .. [1] Takamichi, Shinnosuke, et al. "A postfilter to modify the modulation
      spectrum in HMM-based speech synthesis." Acoustics, Speech and Signal
      Processing (ICASSP), 2014 IEEE International Conference on. IEEE, 2014.

    .. warning::

        This may move in different module in future.

    Args:
        y (numpy.ndarray): Parameter trajectory, shape (``T x D``).
        n (int): DFT length
        norm (str): Normalization mode. See :func:`numpy.fft.fft`.
        return_phase (bool): If True, return phase of MS.

    Returns:
        tuple or numpy.ndarray: Modulation spectrum (``n//2 + 1 x D``) and
        phase (if ``return_phase`` is True).

    See also:
        :func:`nnmnkwii.preprocessing.inv_modspec`,
        :func:`nnmnkwii.autograd.modspec`

    Examples:
        >>> import numpy as np
        >>> from nnmnkwii import preprocessing as P
        >>> generated = np.random.rand(10, 2)
        >>> ms = P.modspec(generated, n=16)
        >>> ms.shape
        (9, 2)
    """
    T, D = x.shape
    # DFT against time axis
    s_complex = np.fft.rfft(x, n=n, axis=0, norm=norm)
    assert s_complex.shape[0] == n // 2 + 1
    R, I = s_complex.real, s_complex.imag
    ms = R * R + I * I

    # TODO: this is ugly...
    if return_phase:
        return ms, np.exp(1.j * np.angle(s_complex))
    else:
        return ms


# For compat
def modphase(x, n=4096, norm=None):
    return modspec(x, n, norm, return_phase=True)[1]


def inv_modspec(ms, phase, norm=None):
    """Inverse transform of modulation spectrum computation

    Given an modulation spectrum and it's phase, it recovers original parameter
    trajectory.

    .. note::
        Returned parameter trajectory has shape (``n x D``), where ``n`` is DFT
        length used in modulation spectrum compuattion. You will have to
        trim it yourself to the actual time length if needed.

    .. warning::

        This may move in different module in future.

    Args:
        ms (numpy.ndarray): Modulation spectrum (``n//2 + 1 x D``).
        phase (numpy.ndarray): Phase of modulation spectrum (``n//2 + 1 x D``).
        norm (str): Normalization mode. See :func:`numpy.fft.fft`.

    Returns:
        numpy.ndarray: Recovered parameter trajectory, shape (``n x D``).

    Examples:
        >>> import numpy as np
        >>> from nnmnkwii import preprocessing as P
        >>> generated = np.random.rand(10, 2)
        >>> ms, phase = P.modspec(generated, n=16, return_phase=True)
        >>> generated_hat = P.inv_modspec(ms, phase)[:len(generated)]
        >>> assert np.allclose(generated, generated_hat)

    See also:
        :func:`nnmnkwii.preprocessing.modspec`.
    """
    n = (ms.shape[0] - 1) * 2

    # |X(x)|^2 -> |X(w)|
    amp = np.sqrt(ms)

    # X(w)
    complex_ms = amp * phase

    # x
    x = np.fft.irfft(complex_ms, n=n, norm=norm, axis=0)
    return x


def modspec_smoothing(x, modfs, n=4096, norm=None, cutoff=50, log_domain=True):
    """Parameter trajectory smoothing by removing high frequency bands of MS.

    Given an parameter trajectory, it removes high frequency bands of its
    modulation spectrum (MS).

    It's known that the effect of the MS components in high MS frequency bands
    on quality of analysis-synthesized speech is negligible in HMM-based speech
    synthesis. See [1]_ for details.

    .. [1] Takamichi, Shinnosuke, et al. "The NAIST text-to-speech system for
      the Blizzard Challenge 2015." Proc. Blizzard Challenge workshop. 2015.

    Args:
        x (numpy.ndarray): Parameter trajectory, shape (``T x D``).
        modfs (int): Sampling frequency in modulation spectrum domain. In
          frame-based processing, this will be ``fs / hop_length``.
        n (int): DFT length
        norm (str): Normalization mode. See :func:`numpy.fft.fft`.
        cutoff (float): Cut-off frequency in Hz.
        log_domain (bool): Whether it performs high frequency band removal on
          log modulation spectrum domain or not.

    Returns:
        numpy.ndarray: Smoothed parameter trajectory, shape (``T x D``).

    Examples:
        >>> import numpy as np
        >>> from nnmnkwii import preprocessing as P
        >>> generated = np.random.rand(10, 2)
        >>> smoothed = P.modspec_smoothing(generated, modfs=200, n=16, cutoff=50)
        >>> smoothed.shape
        (10, 2)
    """
    T, D = x.shape
    if cutoff > modfs // 2:
        raise ValueError(
            "Cutoff frequency {} hz must be larger than Nyquist freqeuency {}. hz".format(
                cutoff, modfs // 2))
    if n < T:
        raise RuntimeError(
            "DFT length {} must be larger than time length {}".format(n, T))

    ms, phase = modspec(x, n=n, norm=norm, return_phase=True)
    if log_domain:
        ms = np.log(ms)

    if cutoff is not None:
        limit_bin = int(n * cutoff / modfs) + 1
        if limit_bin < len(ms):
            ms[limit_bin:] = 0

    if log_domain:
        ms = np.exp(ms)

    x_hat = inv_modspec(ms, phase, norm=norm)
    return np.ascontiguousarray(x_hat[:T])
