from __future__ import division, print_function, absolute_import

import pysptk
import numpy as np
from nnmnkwii.core import VariableLengthArrayTransfomer


class MelCepstrumTransformer(VariableLengthArrayTransfomer):
    def __init__(self, order=24, alpha=0.41, fftlen=512):
        self.order = order
        self.alpha = alpha
        self.fftlen = fftlen

    def do_transform(self, X):
        return np.apply_along_axis(pysptk.sp2mc, 1, X, order=self.order, alpha=self.alpha)

    def do_inverse_transform(self, X):
        return np.apply_along_axis(pysptk.mc2sp, 1, X, alpha=self.alpha, fftlen=self.fftlen)
