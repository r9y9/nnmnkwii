from __future__ import division, print_function, absolute_import


# Note: Tests in the file tightly depends on MLPG and bandmat

from nnmnkwii.autograd._impl.mlpg import MLPG, UnitVarianceMLPG
from nnmnkwii.autograd._impl.modspec import ModSpec
from nnmnkwii import paramgen as G
from nnmnkwii import autograd as AF

from torch.autograd import gradcheck
from torch import nn
import torch
import numpy as np
from warnings import warn

from nose.plugins.attrib import attr


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


def test_functional_mlpg():
    static_dim = 2
    T = 5

    for windows in _get_windows_set():
        torch.manual_seed(1234)
        means = torch.rand(T, static_dim * len(windows))
        variances = torch.ones(static_dim * len(windows))

        y = G.mlpg(means.numpy(), variances.numpy(), windows)
        y = torch.from_numpy(y)

        means = means.clone()
        means.requires_grad = True

        # mlpg
        y_hat = AF.mlpg(means, variances, windows)
        assert np.allclose(y.data.numpy(), y_hat.data.numpy())

        # Test backward pass
        nn.MSELoss()(y_hat, y).backward()

        # unit_variance_mlpg
        R = torch.from_numpy(G.unit_variance_mlpg_matrix(windows, T))
        y_hat = AF.unit_variance_mlpg(R, means)
        assert np.allclose(y.data.numpy(), y_hat.data.numpy())

        nn.MSELoss()(y_hat, y).backward()

        # Test 3D tensor inputs
        y_hat = AF.unit_variance_mlpg(R, means.view(1, -1, means.size(-1)))
        assert np.allclose(
            y.data.numpy(), y_hat.data.view(-1, static_dim).numpy())

        nn.MSELoss()(y_hat.view(-1, static_dim), y).backward()


def test_unit_variance_mlpg_gradcheck():
    static_dim = 2
    T = 10

    for windows in _get_windows_set():
        torch.manual_seed(1234)
        # Meens, input for MLPG
        means = torch.rand(T, static_dim * len(windows), requires_grad=True)

        # Input for UnitVarianceMLPG
        reshaped_means = G.reshape_means(
            means.data.clone().numpy(), static_dim)
        reshaped_means = torch.from_numpy(reshaped_means)
        reshaped_means.requires_grad = True

        # Compute MLPG matrix
        R = G.unit_variance_mlpg_matrix(windows, T).astype(np.float32)
        R = torch.from_numpy(R)

        # UnitVarianceMLPG can take input with both means and reshaped_means
        y1 = UnitVarianceMLPG.apply(means, R)
        y2 = UnitVarianceMLPG.apply(reshaped_means, R)

        # Unit variances
        variances = torch.ones(static_dim * len(windows)
                               ).expand(T, static_dim * len(windows))
        y_hat = MLPG.apply(means, variances, windows)

        # Make sure UnitVarianceMLPG and MLPG can get same result
        # if we use unit variances
        for y in [y1, y2]:
            assert np.allclose(y.data.numpy(), y_hat.data.numpy())

        # Grad check
        inputs = (reshaped_means, R)
        assert gradcheck(UnitVarianceMLPG.apply,
                         inputs, eps=1e-3, atol=1e-3)

        inputs = (means, R)
        assert gradcheck(UnitVarianceMLPG.apply,
                         inputs, eps=1e-3, atol=1e-3)


def test_minibatch_unit_variance_mlpg_gradcheck():
    static_dim = 2
    T = 5

    for windows in _get_windows_set():
        batch_size = 5
        torch.manual_seed(1234)

        # Prepare inputs
        means = torch.rand(T, static_dim * len(windows))
        means_expanded = means.expand(
            batch_size, means.shape[0], means.shape[1])
        reshaped_means = torch.from_numpy(
            G.reshape_means(means.numpy(), static_dim))
        reshaped_means_expanded = reshaped_means.expand(
            batch_size, reshaped_means.shape[0], reshaped_means.shape[1])

        # Target
        y = G.mlpg(means.numpy(), np.ones(static_dim * len(windows)), windows)
        y = torch.from_numpy(y)
        y_expanded = y.expand(batch_size, y.size(0), y.size(1))

        means.requires_grad = True
        means_expanded.requires_grad = True
        reshaped_means.requires_grad = True
        reshaped_means_expanded.requires_grad = True

        # Case 1: 2d with reshaped means
        R = torch.from_numpy(G.unit_variance_mlpg_matrix(windows, T))
        y_hat1 = AF.unit_variance_mlpg(R, reshaped_means)

        # Case 2: 3d with reshaped means
        y_hat2 = AF.unit_variance_mlpg(R, reshaped_means_expanded)
        for i in range(batch_size):
            assert np.allclose(y_hat1.data.numpy(), y_hat2[i].data.numpy())

        nn.MSELoss()(y_hat1, y).backward()
        nn.MSELoss()(y_hat2, y_expanded).backward()

        # Check grad consistency
        for i in range(batch_size):
            grad1 = reshaped_means.grad.data.numpy()
            grad2 = reshaped_means_expanded.grad[i].data.numpy()
            assert np.allclose(grad1, grad2, atol=1.05e-08)

        # Case 3: 2d with non-reshaped input
        y_hat3 = AF.unit_variance_mlpg(R, means)

        # Case 4: 3d with non-reshaped input
        y_hat4 = AF.unit_variance_mlpg(R, means_expanded)

        for i in range(batch_size):
            assert np.allclose(y_hat1.data.numpy(), y_hat3.data.numpy())
            assert np.allclose(y_hat3.data.numpy(), y_hat4[i].data.numpy())

        nn.MSELoss()(y_hat3, y).backward()
        nn.MSELoss()(y_hat4, y_expanded).backward()

        # Check grad consistency
        for i in range(batch_size):
            grad1 = means.grad.data.numpy()
            grad2 = means_expanded.grad[i].data.numpy()
            assert np.allclose(grad1, grad2, atol=1.05e-08)


def test_mlpg_gradcheck():
    # MLPG is performed dimention by dimention, so static_dim 1 is enough,
    # 2 just for in case.
    static_dim = 2
    T = 10

    for windows in _get_windows_set():
        torch.manual_seed(1234)
        means = torch.rand(T, static_dim * len(windows), requires_grad=True)

        # Unit variances case
        variances = torch.ones(static_dim * len(windows)
                               ).expand(T, static_dim * len(windows))
        inputs = (means, variances, windows)

        assert gradcheck(MLPG.apply,
                         inputs, eps=1e-3, atol=1e-3)

        # Rand variances case
        variances = torch.rand(static_dim * len(windows)
                               ).expand(T, static_dim * len(windows))
        inputs = (means, variances, windows)

        assert gradcheck(MLPG.apply,
                         inputs, eps=1e-3, atol=1e-3)


def test_mlpg_variance_expand():
    static_dim = 2
    T = 10

    for windows in _get_windows_set():
        torch.manual_seed(1234)
        means = torch.rand(T, static_dim * len(windows), requires_grad=True)
        variances = torch.rand(static_dim * len(windows))
        variances_expanded = variances.expand(T, static_dim * len(windows))
        y = AF.mlpg(means, variances, windows)
        y_hat = AF.mlpg(means, variances_expanded, windows)
        assert np.allclose(y.data.numpy(), y_hat.data.numpy())


@attr("modspec")
def test_modspec_gradcheck():
    static_dim = 12
    T = 16
    torch.manual_seed(1234)
    n = 16
    for norm in [None, "ortho"]:
        inputs = (torch.rand(T, static_dim, requires_grad=True), n, norm)
        assert gradcheck(ModSpec.apply, inputs, eps=1e-4, atol=1e-4)


@attr("modspec")
def test_modspec_gradcheck_large_n():
    static_dim = 12
    T = 16
    torch.manual_seed(1234)
    for n in [16, 32]:
        for norm in [None, "ortho"]:
            inputs = (torch.rand(T, static_dim, requires_grad=True), n, norm)
            assert gradcheck(ModSpec.apply, inputs, eps=1e-4, atol=1e-4)
