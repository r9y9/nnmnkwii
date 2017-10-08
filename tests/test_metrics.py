# coding: utf-8

import numpy as np
import torch

from nnmnkwii import metrics
from nnmnkwii.util import example_file_data_sources_for_acoustic_model
from nnmnkwii.datasets import FileSourceDataset


def test_mse_variants():
    def __test(f):
        np.random.seed(1234)
        x = np.random.rand(100, 24)
        y = x.copy()
        # 2d case
        assert f(x, y) == 0

        # 1d case
        assert f(np.ones(24), np.ones(24)) == 0

        # batch case
        # should accept torch and numpy arrays
        x1 = np.random.rand(32, 100, 24)
        y1 = np.random.rand(32, 100, 24)
        x2 = torch.rand(32, 100, 24)
        y2 = torch.rand(32, 100, 24)

        for x, y in [(x1, y1), (x2, y2)]:
            lengths = [x.shape[1]] * len(x)
            np.testing.assert_almost_equal(
                f(x, y, lengths), f(x, y), decimal=5)
            assert f(x, y) > 0

    yield __test, metrics.melcd
    yield __test, metrics.mean_squared_error


def test_f0_mse():
    np.random.seed(1234)
    T = 100
    x = np.random.rand(T, 1)
    y = x.copy()

    x_vuv = np.hstack((np.zeros(2), np.ones(T - 2)))
    y_vuv = np.hstack((np.ones(T - 2), np.zeros(2)))

    assert metrics.lf0_mean_squared_error(x, x_vuv, y, y_vuv) == 0
    assert metrics.lf0_mean_squared_error(x, x_vuv, y, y_vuv) == 0

    # batch
    x1 = np.random.rand(32, T, 1)
    y1 = np.random.rand(32, T, 1)
    x1_vuv = np.tile(x_vuv, (32, 1))
    y1_vuv = np.tile(x_vuv, (32, 1))
    x2 = torch.rand(32, T, 1)
    y2 = torch.rand(32, T, 1)
    x2_vuv = torch.from_numpy(x1_vuv).clone()
    y2_vuv = torch.from_numpy(y1_vuv).clone()

    f = metrics.lf0_mean_squared_error
    for linear_domain in [True, False]:
        for x, x_vuv, y, y_vuv in [(x1, x1_vuv, y1, y1_vuv),
                                   (x2, x2_vuv, y2, y2_vuv)]:
            lengths = [x.shape[1]] * len(x)
            np.testing.assert_almost_equal(
                f(x, x_vuv, y, y_vuv, lengths, linear_domain=linear_domain),
                f(x, x_vuv, y, y_vuv, linear_domain=linear_domain), decimal=5)
            assert f(x, x_vuv, y, y_vuv, linear_domain=linear_domain) > 0


def test_vuv_error():
    np.random.seed(1234)
    T = 10
    x_vuv = np.ones(T)
    y_vuv = np.ones(T)

    assert metrics.vuv_error(x_vuv, y_vuv) == 0

    x_vuv = np.hstack((np.ones(T // 2), np.zeros(T // 2)))
    y_vuv = np.hstack((np.zeros(T // 2), np.ones(T // 2)))
    assert metrics.vuv_error(x_vuv, y_vuv) == 1.0

    # 60% overlap, should result in 40% error
    x_vuv = np.hstack((np.ones(int(T * 0.2)), np.zeros(int(T - T * 0.2))))
    y_vuv = np.hstack((np.zeros(int(T - T * 0.2)), np.ones(int(T * 0.2))))
    assert metrics.vuv_error(x_vuv, y_vuv) == 0.4

    # batch
    # should accept torch and numpy arrays
    x1 = np.random.randint(0, 2, (32, 100, 1))
    y1 = np.random.randint(0, 2, (32, 100, 1))
    x2 = torch.from_numpy(x1).clone()
    y2 = torch.from_numpy(y1).clone()

    f = metrics.vuv_error

    for x, y in [(x1, y1), (x2, y2)]:
        lengths = [x.shape[1]] * len(x)
        np.testing.assert_almost_equal(
            f(x, y, lengths), f(x, y), decimal=5)
        assert f(x, y) > 0


def test_real_metrics():
    _, source = example_file_data_sources_for_acoustic_model()
    X = FileSourceDataset(source)
    lengths = [len(x) for x in X]
    X = X.asarray()

    mgc = X[:, :, :source.mgc_dim // 3]
    lf0 = X[:, :, source.lf0_start_idx]
    vuv = (X[:, :, source.vuv_start_idx] > 0).astype(np.int)
    bap = X[:, :, source.bap_start_idx]

    mgc_tgt = mgc + 0.01
    lf0_tgt = lf0 + 0.01
    vuv_tgt = vuv.copy()
    bap_tgt = bap + 0.01

    mcd = metrics.melcd(mgc, mgc_tgt, lengths)
    bap_mcd = metrics.melcd(bap, bap_tgt, lengths)
    lf0_mse = metrics.lf0_mean_squared_error(
        lf0, vuv, lf0_tgt, vuv_tgt, lengths)
    vuv_err = metrics.vuv_error(vuv, vuv_tgt)
    assert mcd > 0
    assert bap_mcd > 0
    assert lf0_mse > 0
    assert vuv_err == 0.0
