from __future__ import division, print_function, absolute_import

import numpy as np
from sklearn.mixture import GaussianMixture
from os.path import join, dirname
from numpy.linalg import norm
from nose.plugins.attrib import attr

DATA_DIR = join(dirname(__file__), "data")


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


@attr("requires_bandmat")
def test_diffvc():
    from nnmnkwii.baseline.gmm import MLPG

    # MLPG is performed dimention by dimention, so static_dim 1 is enough, 2 just for in
    # case.
    static_dim = 2
    T = 10

    for windows in _get_windows_set():
        np.random.seed(1234)
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

        src_mc = src_mc[:, :static_dim]
        tgt_mc = tgt_mc[:, :static_dim]
        assert norm(tgt_mc - mc_converted1) < norm(src_mc - mc_converted1)


@attr("requires_bandmat")
def test_gmmmap_swap():
    from nnmnkwii.baseline.gmm import MLPG

    static_dim = 2
    T = 10
    windows = _get_windows_set()[-1]

    np.random.seed(1234)
    src_mc = np.random.rand(T, static_dim * len(windows))
    tgt_mc = np.random.rand(T, static_dim * len(windows))

    # pseudo parallel data
    XY = np.concatenate((src_mc, tgt_mc), axis=-1)
    gmm = GaussianMixture(n_components=4)
    gmm.fit(XY)

    paramgen = MLPG(gmm, windows=windows, swap=False)
    swap_paramgen = MLPG(gmm, windows=windows, swap=True)

    mc_converted1 = paramgen.transform(src_mc)
    mc_converted2 = swap_paramgen.transform(tgt_mc)

    src_mc = src_mc[:, :static_dim]
    tgt_mc = tgt_mc[:, :static_dim]

    assert norm(tgt_mc - mc_converted1) < norm(src_mc - mc_converted1)
    assert norm(tgt_mc - mc_converted2) > norm(src_mc - mc_converted2)
