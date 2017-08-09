from __future__ import division, print_function, absolute_import

from nnmnkwii.baseline.gmm import MLPG
import numpy as np
from sklearn.mixture import GaussianMixture
from os.path import join, dirname

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
