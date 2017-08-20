# coding: utf-8
from __future__ import with_statement, print_function, absolute_import

from nnmnkwii import functions as F

from torch.autograd import Function
import torch
import numpy as np


class MLPG(Function):
    """Generic MLPG as an autograd function.

    ``f : (T, D) -> (T, static_dim)``.

    This is meant to be used for Minimum Geneartion Error (MGE) training for
    speech synthesis and voice conversion. See [1]_ and [2]_ for details.

    It relies on :func:`nnmnkwii.functions.mlpg` and
    :func:`nnmnkwii.functions.mlpg_grad` for forward and backward computation,
    respectively.

    .. [1] Wu, Zhizheng, and Simon King. "Minimum trajectory error training
      for deep neural networks, combined with stacked bottleneck features."
      INTERSPEECH. 2015.
    .. [2] Xie, Feng-Long, et al. "Sequence error (SE) minimization training of
       neural network for voice conversion." Fifteenth Annual Conference of the
       International Speech Communication Association. 2014.

    Attributes:
        variance_frames (torch.FloatTensor): Variances same as in
            :func:`nnmnkwii.functions.mlpg`.
        windows (list): same as in :func:`nnmnkwii.functions.mlpg`.

    Warnings:
        The function is generic but cannot run on CUDA. For faster
        differenciable MLPG, see :obj:`UnitVarianceMLPG`.

    See also:
        :func:`nnmnkwii.autograd.mlpg`,
        :func:`nnmnkwii.functions.mlpg`,
        :func:`nnmnkwii.functions.mlpg_grad`.
    """

    def __init__(self, variance_frames, windows):
        super(MLPG, self).__init__()
        self.windows = windows
        self.variance_frames = variance_frames

    def forward(self, mean_frames):
        assert mean_frames.dim() == 2  # we cannot do MLPG on minibatch
        variance_frames = self.variance_frames
        self.save_for_backward(mean_frames)

        T, D = mean_frames.size()
        assert mean_frames.size() == variance_frames.size()

        mean_frames_np = mean_frames.numpy()
        variance_frames_np = variance_frames.numpy()
        y = F.mlpg(mean_frames_np, variance_frames_np, self.windows)
        y = torch.from_numpy(y.astype(np.float32))
        return y

    def backward(self, grad_output):
        mean_frames, = self.saved_tensors
        variance_frames = self.variance_frames

        T, D = mean_frames.size()

        grad_output_numpy = grad_output.numpy()
        mean_frames_numpy = mean_frames.numpy()
        variance_frames_numpy = variance_frames.numpy()
        grads_numpy = F.mlpg_grad(
            mean_frames_numpy, variance_frames_numpy, self.windows,
            grad_output_numpy)

        return torch.from_numpy(grads_numpy).clone()


class UnitVarianceMLPG(Function):
    """Special case of MLPG assuming data is normalized to have unit variance.

    ``f : (T*num_windows, static_dim) -> (T, static_dim)``.

    The funtion is theoretically a special case of :obj:`MLPG`. The function
    assumes input data is noramlized to have unit variance for each dimention.
    The property of the unit-variance greatly simplifies the backward
    computation of MLPG.

    Let :math:`\mu` is the input mean sequence (``num_windows*T x static_dim``),
    :math:`W` is a window matrix ``(T x num_windows*T)``, MLPG can be written
    as follows:

    .. math::

        y = R \mu

    where

    .. math::

        R = (W^{T} W)^{-1} W^{T}

    Note that we offen represent static + dynamic features as
    (``T x static_dim*num_windows``) matirx, but the function assumes input has
    shape of (``num_windows*T x static_dim``).

    To avoid dupulicate computations in forward and backward, the function
    takes ``R`` at construction time. The matrix ``R`` can be computed by
    :func:`nnmnkwii.functions.unit_variance_mlpg_matrix`.

    Args:
        R: Unit-variance MLPG matrix of shape (``T x num_windows*T``). This
          should be created with
          :func:`nnmnkwii.functions.unit_variance_mlpg_matrix`.


    Attributes:
        R: Unit-variance MLPG matrix (``T x num_windows*T``).

    See also:
        :func:`nnmnkwii.autograd.unit_variance_mlpg`.
    """
    def __init__(self, R):
        super(UnitVarianceMLPG, self).__init__()
        self.R = R

    def forward(self, means):
        return torch.mm(self.R, means)

    def backward(self, grad_output):
        return torch.mm(self.R.transpose(0,1), grad_output)


def mlpg(mean_frames, variance_frames, windows):
    """Maximum Liklihood Paramter Generation (MLPG).

    The parameters are almost same as :func:`nnmnkwii.functions.mlpg` expects.
    The differences are:

    - The function assumes ``mean_frames`` as :obj:`torch.autograd.Variable`
      instead of :obj:`numpy.ndarray`.
    - The fucntion assumes ``variances_frames`` as :obj:`torch.FloatTensor`　
      instead of :obj:`numpy.ndarray`.

    Args:
        mean_frames (torch.autograd.Variable): Means
        variance_frames (torch.FloatTensor): Variances
        windows (list): A sequence of window specification

    See also:
        :obj:`nnmnkwii.autograd.MLPG`, :func:`nnmnkwii.functions.mlpg`

    """
    T, D = mean_frames.size()
    if variance_frames.dim() == 1 and variance_frames.shape[0] == D:
        variance_frames = variance_frames.expand(T, D)
    assert mean_frames.size() == variance_frames.size()
    return MLPG(variance_frames, windows)(mean_frames)

def unit_variance_mlpg(R, means):
    """Special case of MLPG assuming data is normalized to have unit variance.

    Args:
        means (torch.autograd.Variable): Means, of shape
          (``T*num_windows x static_dim``). See
          :func:`nnmnkwii.functions.reshape_means` to reshape means from
          (``T x D``) to (``T*num_windows x static_dim``).
        R (torch.FloatTensor): MLPG matrix.

    See also:
        :obj:`nnmnkwii.autograd.UnitVarianceMLPG`,
        :func:`nnmnkwii.functions.unit_variance_mlpg_matrix`,
        :func:`reshape_means`.
    """
    return UnitVarianceMLPG(R)(means)
