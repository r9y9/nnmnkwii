from __future__ import division, print_function, absolute_import

from nnmnkwii.preprocessing.f0 import interp1d
import numpy as np


def test_interp1d():
    f0 = np.random.randint(0, 10, 100).astype(np.float32)
    if0 = interp1d(f0)
    # TODO: better test
    assert np.all(np.isfinite(if0))
