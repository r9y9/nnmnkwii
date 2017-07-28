from nnmnkwii.autograd._impl.mlpg import MLPG

from torch.autograd import gradcheck
from torch.autograd import Variable
import torch
import numpy as np


def test_mlpg_gradcheck():
    static_dim = 24
    T = 10
    windows = [
        (0, 0, np.array([1.0])),
        # TODO: maybe we need to be careful abount edges? (t=0, t=T-1)
        # (1, 1, np.array([-0.5, 0.0, 0.5])),
    ]
    means = Variable(torch.rand(T, static_dim * len(windows)),
                     requires_grad=True)
    variances = torch.rand(static_dim * len(windows)
                           ).expand(T, static_dim * len(windows))

    inputs = (means,)

    assert gradcheck(MLPG(static_dim, variances, windows), inputs, eps=1e-4)
