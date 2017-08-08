from __future__ import division, print_function, absolute_import

from nnmnkwii.baseline.gmm import MLPG
import numpy as np
import pysptk
from sklearn.mixture import GaussianMixture
from os.path import join, dirname

from nnmnkwii.baseline.post_filters import merlin_post_filter

DATA_DIR = join(dirname(__file__), "data")


def test_diffvc():
    # MLPG is performed dimention by dimention, so static_dim 1 is enough, 2 just for in
    # case.
    static_dim = 2
    T = 10

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

    for windows in windows_set:
        src_mc = np.random.rand(T, static_dim * len(windows))
        tgt_mc = np.random.rand(T, static_dim * len(windows))

        # pseudo parallel data
        XY = np.concatenate((src_mc, tgt_mc), axis=-1)
        gmm = GaussianMixture(n_components=4)
        gmm.fit(XY)

        paramgen = MLPG(gmm, windows=windows, diff=False)
        diff_paramgen = MLPG(gmm, windows=windows, diff=True)

        mc_converted1 = paramgen.transform(src_mc)
        mc_converted2 = diff_paramgen.transform(src_mc)

        assert mc_converted1.shape == (T, static_dim)
        assert mc_converted2.shape == (T, static_dim)


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
