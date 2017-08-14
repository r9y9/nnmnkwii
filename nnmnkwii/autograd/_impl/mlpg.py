from __future__ import with_statement, print_function, absolute_import

from nnmnkwii import functions as F

# TODO: should this be exported?
from nnmnkwii.functions._impl.mlpg import build_win_mats

from torch.autograd import Function
import torch
import numpy as np
import bandmat as bm


# Note: this is written for pytorch 0.1.12 and may not compatible with
# pytorch master.
class MLPG(Function):
    """MLPG as an autograd function ``f : (T, D) -> (T, static_dim)``.

    This is meant to be used for Minimum Geneartion Error (MGE) training for
    speech synthesis and voice conversion. See [1]_ for details.

    .. [1] Wu, Zhizheng, and Simon King. "Minimum trajectory error training
      for deep neural networks, combined with stacked bottleneck features."
      INTERSPEECH. 2015.

    Let :math:`d` is the index of static features, :math:`l` is the index
    of windows, gradients :math:`g_{d,l}` can be computed by:

    .. math::

        g_{d,l} = (\sum_{l} W_{l}^{T}P_{d,l}W_{l})^{-1} W_{l}^{T}P_{d,l}

    where :math:`W_{l}` is a banded window matrix and :math:`P_{d,l}` is a
    diagonal precision matrix.

    Assuming the variances are diagonals, MLPG can be performed in
    dimention-by-dimention efficiently.

    Let :math:`o_{d}` be ``T`` dimentional back-propagated gradients, the
    resulting gradients :math:`g'_{l,d}` to be propagated are
    computed as follows:

    .. math::

        g'_{d,l} = o_{d}^{T} g_{d,l}

    Attributes:
        static_dim (int): number of static dimentions
        variance_frames (torch.FloatTensor): Variances same as in
            :func:`nnmnkwii.functions.mlpg`.
        windows (list): same as in :func:`nnmnkwii.functions.mlpg`.

    TODO:
        CUDA implementation

    See also:
        :func:`nnmnkwii.functions.mlpg`.
    """

    def __init__(self, static_dim, variance_frames, windows):
        super(MLPG, self).__init__()
        self.static_dim = static_dim
        self.windows = windows
        self.variance_frames = variance_frames

    def forward(self, mean_frames):
        assert mean_frames.dim() == 2  # we cannot do MLPG on minibatch
        variance_frames = self.variance_frames
        self.save_for_backward(mean_frames)

        T, D = mean_frames.size()
        assert mean_frames.size() == variance_frames.size()
        assert self.static_dim == D // len(self.windows)

        mean_frames_np = mean_frames.numpy()
        variance_frames_np = variance_frames.numpy()
        y = F.mlpg(mean_frames_np, variance_frames_np, self.windows)
        y = torch.from_numpy(y.astype(np.float32))
        return y

    def backward(self, grad_output):
        mean_frames, = self.saved_tensors
        variance_frames = self.variance_frames

        T, D = mean_frames.size()
        win_mats = build_win_mats(self.windows, T)

        grads = torch.zeros(T, D)
        for d in range(self.static_dim):
            sdw = max([win_mat.l + win_mat.u for win_mat in win_mats])

            # R: \sum_{l} W_{l}^{T}P_{d,l}W_{l}
            R = bm.zeros(sdw, sdw, T)  # overwritten in the loop

            # dtype = np.float64 for bandmat
            precisions = np.zeros((len(self.windows), T), dtype=np.float64)

            for win_idx, win_mat in enumerate(win_mats):
                precisions[win_idx] = 1 / \
                    variance_frames[:, win_idx * self.static_dim + d].numpy()

                bm.dot_mm_plus_equals(win_mat.T, win_mat,
                                      target_bm=R, diag=precisions[win_idx])

            for win_idx, win_mat in enumerate(win_mats):
                # r: W_{l}^{T}P_{d,l}
                r = bm.dot_mm(win_mat.T, bm.diag(precisions[win_idx]))

                # grad_{d, l} = R^{-1r}
                grad = np.linalg.solve(R.full(), r.full())
                assert grad.shape == (T, T)

                # Finally we get grad for a particular dimention
                grads[:, win_idx * self.static_dim +
                      d] = torch.from_numpy(grad_output[:, d].numpy().T.dot(grad))

        return grads


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
        :func:`nnmnkwii.functions.mlpg`

    """
    T, D = mean_frames.size()
    assert mean_frames.size() == variance_frames.size()
    static_dim = D // len(windows)
    return MLPG(static_dim, variance_frames, windows)(mean_frames)
