# coding: utf-8
from __future__ import with_statement, print_function, absolute_import

from nnmnkwii import paramgen as G

from torch.autograd import Function
import torch
import numpy as np


class MLPG(Function):
    """Generic MLPG as an autograd function.

    ``f : (T, D) -> (T, static_dim)``.

    This is meant to be used for Minimum Geneartion Error (MGE) training for
    speech synthesis and voice conversion. See [1]_ and [2]_ for details.

    It relies on :func:`nnmnkwii.paramgen.mlpg` and
    :func:`nnmnkwii.paramgen.mlpg_grad` for forward and backward computation,
    respectively.

    .. [1] Wu, Zhizheng, and Simon King. "Minimum trajectory error training
      for deep neural networks, combined with stacked bottleneck features."
      INTERSPEECH. 2015.
    .. [2] Xie, Feng-Long, et al. "Sequence error (SE) minimization training of
       neural network for voice conversion." Fifteenth Annual Conference of the
       International Speech Communication Association. 2014.

    Args:
        variances (torch.FloatTensor): Variances same as in
            :func:`nnmnkwii.paramgen.mlpg`.
        windows (list): same as in :func:`nnmnkwii.paramgen.mlpg`.

    Warnings:
        The function is generic but cannot run on CUDA. For faster
        differenciable MLPG, see :obj:`UnitVarianceMLPG`.

    See also:
        :func:`nnmnkwii.autograd.mlpg`,
        :func:`nnmnkwii.paramgen.mlpg`,
        :func:`nnmnkwii.paramgen.mlpg_grad`.
    """

    @staticmethod
    def forward(ctx, means, variances, windows):
        assert means.dim() == 2  # we cannot do MLPG on minibatch
        ctx.windows = windows
        ctx.save_for_backward(means, variances)

        T, D = means.size()
        assert means.size() == variances.size()

        means_np = means.detach().numpy()
        variances_np = variances.detach().numpy()
        y = G.mlpg(means_np, variances_np, ctx.windows)
        y = torch.from_numpy(y.astype(np.float32))
        return y

    @staticmethod
    def backward(ctx, grad_output):
        means, variances = ctx.saved_tensors

        T, D = means.size()

        grad_output_numpy = grad_output.detach().numpy()
        means_numpy = means.detach().numpy()
        variances_numpy = variances.detach().numpy()
        grads_numpy = G.mlpg_grad(
            means_numpy, variances_numpy, ctx.windows,
            grad_output_numpy)

        return torch.from_numpy(grads_numpy).clone(), None, None


class UnitVarianceMLPG(Function):
    """Special case of MLPG assuming data is normalized to have unit variance.

    ``f : (T x D) -> (T, static_dim)``. or
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

    The matrix ``R`` can be computed by
    :func:`nnmnkwii.paramgen.unit_variance_mlpg_matrix`.

    Args:
        R: Unit-variance MLPG matrix of shape (``T x num_windows*T``). This
          should be created with
          :func:`nnmnkwii.paramgen.unit_variance_mlpg_matrix`.

    See also:
        :func:`nnmnkwii.autograd.unit_variance_mlpg`.
    """

    @staticmethod
    def forward(ctx, means, R):
        # TODO: remove this
        ctx.save_for_backward(means, R)
        ctx.num_windows = R.shape[-1] // R.shape[0]
        T = R.shape[0]
        dim = means.dim()

        # Add batch axis if necessary
        if dim == 2:
            T_, D = means.shape
            B = 1
            means = means.view(B, T_, D)
        else:
            B, T_, D = means.shape

        # Check if means has proper shape
        reshaped = not (T == T_)
        if not reshaped:
            static_dim = means.shape[-1] // ctx.num_windows
            reshaped_means = means.contiguous().view(
                B, T, ctx.num_windows, -1).transpose(
                    1, 2).contiguous().view(B, -1, static_dim)
        else:
            static_dim = means.shape[-1]
            reshaped_means = means

        out = torch.matmul(R, reshaped_means)
        if dim == 2:
            return out.view(-1, static_dim)

        return out

    @staticmethod
    def backward(ctx, grad_output):
        means, R = ctx.saved_tensors
        T = R.shape[0]
        dim = means.dim()

        # Add batch axis if necessary
        if dim == 2:
            T_, D = means.shape
            B = 1
            grad_output = grad_output.view(B, T, -1)
        else:
            B, T_, D = means.shape

        grad = torch.matmul(R.transpose(0, 1), grad_output)

        reshaped = not (T == T_)
        if not reshaped:
            grad = grad.view(B, ctx.num_windows, T, -1).transpose(
                1, 2).contiguous().view(B, T, D)

        if dim == 2:
            return grad.view(-1, D), None

        return grad, None


def mlpg(means, variances, windows):
    """Maximum Liklihood Paramter Generation (MLPG).

    The parameters are almost same as :func:`nnmnkwii.paramgen.mlpg` expects.
    The differences are:

    - The function assumes ``means`` as :obj:`torch.autograd.Variable`
      instead of :obj:`numpy.ndarray`.
    - The fucntion assumes ``variances_frames`` as :obj:`torch.FloatTensor`　
      instead of :obj:`numpy.ndarray`.

    Args:
        means (torch.autograd.Variable): Means
        variances (torch.FloatTensor): Variances
        windows (list): A sequence of window specification

    See also:
        :obj:`nnmnkwii.autograd.MLPG`, :func:`nnmnkwii.paramgen.mlpg`

    """
    T, D = means.size()
    if variances.dim() == 1 and variances.shape[0] == D:
        variances = variances.expand(T, D)
    assert means.size() == variances.size()
    return MLPG.apply(means, variances, windows)


def unit_variance_mlpg(R, means):
    """Special case of MLPG assuming data is normalized to have unit variance.

    Args:
        means (torch.autograd.Variable): Means, of shape (``T x D``) or
          (``T*num_windows x static_dim``). See
          :func:`nnmnkwii.paramgen.reshape_means` to reshape means from
          (``T x D``) to (``T*num_windows x static_dim``).
        R (torch.FloatTensor): MLPG matrix.

    See also:
        :obj:`nnmnkwii.autograd.UnitVarianceMLPG`,
        :func:`nnmnkwii.paramgen.unit_variance_mlpg_matrix`,
        :func:`reshape_means`.
    """
    return UnitVarianceMLPG.apply(means, R)
