from __future__ import division, print_function, absolute_import

from nnmnkwii import functions as F
from nnmnkwii import autograd as AF
from torch.autograd import Variable
import torch
from torch import nn
import numpy as np
import time
import sys


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


OKGREEN = '\033[92m'
FAIL = '\033[91m'
ENDC = '\033[0m'


def benchmark_mlpg(static_dim=59, T=100, batch_size=10, use_cuda=True):
    if use_cuda and not torch.cuda.is_available():
        return

    windows = _get_windows_set()[-1]
    np.random.seed(1234)
    torch.manual_seed(1234)
    means = np.random.rand(T, static_dim * len(windows)).astype(np.float32)
    variances = np.ones(static_dim * len(windows))
    reshaped_means = F.reshape_means(means, static_dim)

    # Ppseud target
    y = F.mlpg(means, variances, windows).astype(np.float32)

    # Pack into variables
    means = Variable(torch.from_numpy(means), requires_grad=True)
    reshaped_means = Variable(
        torch.from_numpy(reshaped_means), requires_grad=True)
    y = Variable(torch.from_numpy(y), requires_grad=False)
    criterion = nn.MSELoss()

    # Case 1: MLPG
    since = time.time()
    for _ in range(batch_size):
        y_hat = AF.mlpg(means, torch.from_numpy(variances), windows)
        L = criterion(y_hat, y)
        assert np.allclose(y_hat.data.numpy(), y.data.numpy())
        L.backward()  # slow!
    elapsed_mlpg = time.time() - since

    # Case 2: UnitVarianceMLPG
    since = time.time()
    if use_cuda:
        y = y.cuda()
    R = F.unit_variance_mlpg_matrix(windows, T)
    R = torch.from_numpy(R)
    # Assuming minibatch are zero-ppaded, we only need to create MLPG matrix
    # per-minibatch, not per-utterance.
    if use_cuda:
        R = R.cuda()
    for _ in range(batch_size):
        if use_cuda:
            means = means.cpu()
            means = means.cuda()

        y_hat = AF.unit_variance_mlpg(R, means)
        L = criterion(y_hat, y)
        assert np.allclose(y_hat.cpu().data.numpy(), y.cpu().data.numpy(),
                           atol=1e-5)
        L.backward()
    elapsed_unit_variance_mlpg = time.time() - since

    ratio = elapsed_mlpg / elapsed_unit_variance_mlpg

    print(
        "MLPG vs UnitVarianceMLPG (static_dim, T, batch_size, use_cuda) = ({}):".format(
            (static_dim, T, batch_size, use_cuda)))
    if ratio > 1:
        s = "faster"
        sys.stdout.write(OKGREEN)
    else:
        s = "slower"
        sys.stdout.write(FAIL)
    print("UnitVarianceMLPG, {:4f} times {}. Elapsed times {:4f} / {:4f}".format(
        ratio, s, elapsed_mlpg, elapsed_unit_variance_mlpg))

    print(ENDC)


if __name__ == "__main__":
    for use_cuda in [False, True]:
        for static_dim in [24, 59]:
            for T in [500, 1000]:
                for batch_size in [1, 5, 10]:
                    benchmark_mlpg(
                        static_dim=static_dim, T=T,
                        batch_size=batch_size, use_cuda=use_cuda)
