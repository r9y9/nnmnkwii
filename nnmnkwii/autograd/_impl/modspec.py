from __future__ import with_statement, print_function, absolute_import

from nnmnkwii.preprocessing.modspec import modspec as _modspec

from torch.autograd import Function
import torch
import numpy as np


class ModSpec(Function):
    """Modulation spectrum computation ``f : (T, D) -> (N//2+1, D)``.

    Args:
        n (int): DFT length.
        norm (bool): Normalize DFT output or not. See :obj:`numpy.fft.fft`.
    """
    @staticmethod
    def forward(ctx, y, n, norm):
        ctx.n = n
        ctx.norm = norm
        assert y.dim() == 2
        ctx.save_for_backward(y)

        y_np = y.detach().numpy()
        ms = torch.from_numpy(_modspec(y_np, n=n, norm=norm))

        return ms

    @staticmethod
    def backward(ctx, grad_output):
        y, = ctx.saved_tensors
        T, D = y.size()
        assert grad_output.size() == torch.Size((ctx.n // 2 + 1, D))

        y_np = y.detach().numpy()
        kt = -2 * np.pi / ctx.n * np.arange(ctx.n // 2 +
                                            1)[:, None] * np.arange(T)

        assert kt.shape == (ctx.n // 2 + 1, T)
        cos_table = np.cos(kt)
        sin_table = np.sin(kt)

        R = np.zeros((ctx.n // 2 + 1, D))
        I = np.zeros((ctx.n // 2 + 1, D))
        s_complex = np.fft.rfft(y_np, n=ctx.n, axis=0,
                                norm=ctx.norm)  # DFT against time axis
        assert s_complex.shape == (ctx.n // 2 + 1, D)
        R, I = s_complex.real, s_complex.imag

        grads = torch.zeros(T, D)
        C = 2  # normalization constant
        if ctx.norm == "ortho":
            C /= np.sqrt(ctx.n)

        for d in range(D):
            r = R[:, d][:, None]
            i = I[:, d][:, None]
            grad = C * (r * cos_table + i * sin_table)
            assert grad.shape == sin_table.shape
            grads[:, d] = torch.from_numpy(
                grad_output[:, d].numpy().T.dot(grad))

        return grads, None, None


def modspec(y, n=2048, norm=None):
    """Moduration spectrum computation.

    Args:
        y (torch.autograd.Variable): Parameter trajectory.
        n (int): DFT length.
        norm (bool): Normalize DFT output or not. See :obj:`numpy.fft.fft`.

    """
    return ModSpec.apply(y, n, norm)
