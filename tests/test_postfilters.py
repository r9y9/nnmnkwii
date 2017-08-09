from __future__ import division, print_function, absolute_import

import numpy as np
import pysptk
from os.path import join, dirname

from nnmnkwii.postfilters import merlin_post_filter

DATA_DIR = join(dirname(__file__), "data")


def test_merlin_post_filter():
    root = join(DATA_DIR, "merlin_post_filter")
    mgc = np.fromfile(join(root, "arctic_b0539.mgc"),
                      dtype=np.float32).reshape(-1, 60)
    weight = np.fromfile(join(root, "weight"), dtype=np.float32)
    alpha = 0.58
    minimum_phase_order = 511
    fftlen = 1024
    coef = 1.4

    # Step 1
    mgc_r0 = np.fromfile(join(root, "arctic_b0539.mgc_r0"), dtype=np.float32)
    mgc_r0_hat = pysptk.c2acr(pysptk.freqt(
        mgc, minimum_phase_order, alpha=-alpha), 0, fftlen).flatten()
    assert np.allclose(mgc_r0, mgc_r0_hat)

    # Step 2
    mgc_p_r0 = np.fromfile(
        join(root, "arctic_b0539.mgc_p_r0"), dtype=np.float32)
    mgc_p_r0_hat = pysptk.c2acr(pysptk.freqt(
        mgc * weight, minimum_phase_order, -alpha), 0, fftlen).flatten()
    assert np.allclose(mgc_p_r0, mgc_p_r0_hat)

    # Step 3
    mgc_b0 = np.fromfile(join(root, "arctic_b0539.mgc_b0"), dtype=np.float32)
    mgc_b0_hat = pysptk.mc2b(weight * mgc, alpha)[:, 0]
    assert np.allclose(mgc_b0, mgc_b0_hat)

    # Step 4
    mgc_p_b0 = np.fromfile(
        join(root, "arctic_b0539.mgc_p_b0"), dtype=np.float32)
    mgc_p_b0_hat = np.log(mgc_r0_hat / mgc_p_r0_hat) / 2 + mgc_b0_hat
    assert np.allclose(mgc_p_b0, mgc_p_b0_hat)

    # Final step
    mgc_p_mgc = np.fromfile(
        join(root, "arctic_b0539.mgc_p_mgc"), dtype=np.float32).reshape(-1, 60)
    mgc_p_mgc_hat = pysptk.b2mc(
        np.hstack((mgc_p_b0_hat[:, None], pysptk.mc2b(mgc * weight, alpha)[:, 1:])), alpha)
    assert np.allclose(mgc_p_mgc, mgc_p_mgc_hat)

    filtered_mgc = merlin_post_filter(mgc, alpha, coef=coef, weight=weight,
                                      minimum_phase_order=minimum_phase_order,
                                      fftlen=fftlen)
    assert np.allclose(filtered_mgc, mgc_p_mgc, atol=1e-6)
