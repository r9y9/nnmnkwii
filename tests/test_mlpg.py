from nnmnkwii.autograd._impl.mlpg import MLPG

from torch.autograd import gradcheck
from torch.autograd import Variable
import torch
import numpy as np


def test_mlpg_gradcheck():
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
        torch.manual_seed(1234)
        means = Variable(torch.rand(T, static_dim * len(windows)),
                         requires_grad=True)
        inputs = (means,)

        # Unit variances case
        variances = torch.ones(static_dim * len(windows)
                               ).expand(T, static_dim * len(windows))

        assert gradcheck(MLPG(static_dim, variances, windows),
                         inputs, eps=1e-3, atol=1e-3)

        # Rand variances case
        variances = torch.rand(static_dim * len(windows)
                               ).expand(T, static_dim * len(windows))

        assert gradcheck(MLPG(static_dim, variances, windows),
                         inputs, eps=1e-3, atol=1e-3)
