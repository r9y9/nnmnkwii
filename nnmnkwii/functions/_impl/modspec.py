from __future__ import with_statement, print_function, absolute_import

import numpy as np


def modspec(y, n=4096, norm=None):
    T, D = y.shape
    # DFT against time axis
    s_complex = np.fft.rfft(y, n=n, axis=0, norm=norm)
    assert s_complex.shape[0] == n // 2 + 1
    R, I = s_complex.real, s_complex.imag
    return R * R + I * I
