from __future__ import with_statement, print_function, absolute_import

import numpy as np


def melcd(vec1, vec2):
    return 10.0 * np.sqrt(2 * np.sum(np.square(vec1 - vec2))) / np.log(10)
