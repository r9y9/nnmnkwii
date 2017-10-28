from __future__ import with_statement, print_function, absolute_import

from nnmnkwii.preprocessing.modspec import modspec as _modspec

from torch.autograd import Function
import torch
import numpy as np


class ModSpec(Function):
    """Modulation spectrum computation ``f : (T, D) -> (N//2+1, D)``.

    Attributes:
        n (int): DFT length.
        norm (bool): Normalize DFT output or not. See :obj:`numpy.fft.fft`.
    """

    def __init__(self, n=2048, norm=None):
        self.n = n
        self.norm = norm

    def forward(self, y):
        assert y.dim() == 2
        self.save_for_backward(y)

        y_np = y.numpy()
        ms = torch.from_numpy(_modspec(y_np, n=self.n, norm=self.norm))

        return ms

    def backward(self, grad_output):
        y, = self.saved_tensors
        T, D = y.size()
        assert grad_output.size() == torch.Size((self.n // 2 + 1, D))

        y_np = y.numpy()
        kt = -2 * np.pi / self.n * np.arange(self.n // 2 +
                                             1)[:, None] * np.arange(T)

        assert kt.shape == (self.n // 2 + 1, T)
        cos_table = np.cos(kt)
        sin_table = np.sin(kt)

        R = np.zeros((self.n // 2 + 1, D))
        I = np.zeros((self.n // 2 + 1, D))
        s_complex = np.fft.rfft(y_np, n=self.n, axis=0,
                                norm=self.norm)  # DFT against time axis
        assert s_complex.shape == (self.n // 2 + 1, D)
        R, I = s_complex.real, s_complex.imag

        grads = torch.zeros(T, D)
        C = 2  # normalization constant
        if self.norm == "ortho":
            C /= np.sqrt(self.n)

        for d in range(D):
            r = R[:, d][:, None]
            i = I[:, d][:, None]
            grad = C * (r * cos_table + i * sin_table)
            assert grad.shape == sin_table.shape
            grads[:, d] = torch.from_numpy(
                grad_output[:, d].numpy().T.dot(grad))

        return grads


def modspec(y, n=2048, norm=None):
    """Moduration spectrum computation.

    Args:
        y (torch.autograd.Variable): Parameter trajectory.
        n (int): DFT length.
        norm (bool): Normalize DFT output or not. See :obj:`numpy.fft.fft`.

    """
    return ModSpec(n=n, norm=norm)(y)
