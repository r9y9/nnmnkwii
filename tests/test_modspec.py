from nnmnkwii.autograd._impl.modspec import ModSpec

from torch.autograd import gradcheck
from torch.autograd import Variable
import torch


def test_modspec_gradcheck():
    static_dim = 12
    T = 16
    inputs = (Variable(torch.rand(T, static_dim), requires_grad=True),)
    n = 16

    for norm in [None, "ortho"]:
        assert gradcheck(ModSpec(n=n, norm=norm), inputs, eps=1e-4, atol=1e-4)

    for n in [16, 32]:
        for norm in [None, "ortho"]:
            assert gradcheck(ModSpec(n=n, norm=norm),
                             inputs, eps=1e-4, atol=1e-4)
