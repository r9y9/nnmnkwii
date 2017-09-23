# coding: utf-8
from __future__ import division, print_function, absolute_import

import pysptk
import numpy as np

__all__ = ['merlin_post_filter']


def merlin_post_filter(mgc, alpha,
                       minimum_phase_order=511, fftlen=1024,
                       coef=1.4, weight=None):
    """Post-filter used in Merlin.

    This is a :obj:`pysptk` translation of `Merlin's post filter`_ written with
    SPTK CLI tools. Details can be found at `CSTR-Edinburgh/merlin/issues/241`_
    and [1]_.

    .. _Merlin's post filter: https://goo.gl/jK5Hdd
    .. _`CSTR-Edinburgh/merlin/issues/241`: https://github.com/CSTR-Edinburgh/merlin/issues/241


    .. [1] Yoshimura, Takayoshi, et al. "Incorporating a mixed excitation model
        and postfilter into HMM‐based text‐to‐speech synthesis." Systems and
        Computers in Japan 36.12 (2005): 43-50.

    Args:
        mgc (2darray): mel-generalized cepstrum
        alpha (float): all-pass constant
        minimum_phase_order (int): Order of minimum phase sequence
        fftlen (int): FFT length used to convert cepstrum to autocorrelation.
        coef (float): Weight coefficient to build weight vector. Default is 1.4.
        weight ([optional]1darray): Weight vector for scaling mel-generalized
            cepstrum. If None, set automatically by ``coef`` as in Merlin.

    Returns:
        2darray: Post-filtered mel-generalized cepstrum.

    Examples:
        >>> from nnmnkwii.postfilters import merlin_post_filter
        >>> import numpy as np
        >>> mgc = np.random.rand(100, 60)
        >>> mgc_filtered = merlin_post_filter(mgc, 0.58)
        >>> assert mgc.shape == mgc_filtered.shape

    """
    _, D = mgc.shape
    if weight is None:
        weight = np.ones(D) * coef
        weight[:2] = 1
    assert len(weight) == D

    mgc_r0 = pysptk.c2acr(pysptk.freqt(
        mgc, minimum_phase_order, alpha=-alpha), 0, fftlen).flatten()
    mgc_p_r0 = pysptk.c2acr(pysptk.freqt(
        mgc * weight, minimum_phase_order, -alpha), 0, fftlen).flatten()
    mgc_b0 = pysptk.mc2b(weight * mgc, alpha)[:, 0]
    mgc_p_b0 = np.log(mgc_r0 / mgc_p_r0) / 2 + mgc_b0
    mgc_p_mgc = pysptk.b2mc(
        np.hstack((mgc_p_b0[:, None], pysptk.mc2b(mgc * weight, alpha)[:, 1:])), alpha)

    return mgc_p_mgc
