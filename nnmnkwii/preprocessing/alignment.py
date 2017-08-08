from __future__ import division, print_function, absolute_import

from nnmnkwii.util import trim_zeros_frames
from nnmnkwii.baseline.gmm import MLPG

from fastdtw import fastdtw

import numpy as np
from numpy.linalg import norm

from sklearn.mixture import GaussianMixture


class DTWAligner(object):
    """Align feature matcies

    Attributes:
        dist (function): Distance function
        radius (int): Radius
        verbose (int): Default is 0
    """

    def __init__(self, dist=lambda x, y: norm(x - y), radius=1, verbose=0):
        self.verbose = verbose
        self.dist = dist
        self.radius = radius

    def transform(self, XY):
        X, Y = XY
        assert X.ndim == 3 and Y.ndim == 3

        X_aligned = np.zeros_like(X)
        Y_aligned = np.zeros_like(Y)
        for idx, (x, y) in enumerate(zip(X, Y)):
            x, y = trim_zeros_frames(x), trim_zeros_frames(y)
            dist, path = fastdtw(x, y, radius=self.radius, dist=self.dist)
            dist /= (len(x) + len(y))
            pathx = list(map(lambda l: l[0], path))
            pathy = list(map(lambda l: l[1], path))
            x, y = x[pathx], y[pathy]
            X_aligned[idx][:len(x)] = x
            Y_aligned[idx][:len(y)] = y
            if self.verbose > 0:
                print("{}, distance: {}".format(idx, dist))
        return X_aligned, Y_aligned


class IterativeDTWAligner(object):
    """Align feature matcies iteratively using GMM-based feature conversion

    Attributes:
        n_iter (int): Number of iterations.
        dist (function): Distance function
        radius (int): Radius
        verbose (int): Default is 0
    """

    def __init__(self, n_iter=3, dist=lambda x, y: norm(x - y), radius=1, verbose=0):
        self.n_iter = n_iter
        self.dist = dist
        self.radius = radius
        self.verbose = verbose

    def transform(self, XY):
        X, Y = XY
        assert X.ndim == 3 and Y.ndim == 3

        Xc = X.copy()  # this will be updated iteratively
        X_aligned = np.zeros_like(X)
        Y_aligned = np.zeros_like(Y)
        refined_paths = np.empty(len(X), dtype=np.object)

        for idx in range(self.n_iter):
            for idx, (x, y) in enumerate(zip(Xc, Y)):
                x, y = trim_zeros_frames(x), trim_zeros_frames(y)
                dist, path = fastdtw(x, y, radius=self.radius, dist=self.dist)
                dist /= (len(x) + len(y))
                pathx = list(map(lambda l: l[0], path))
                pathy = list(map(lambda l: l[1], path))

                refined_paths[idx] = pathx
                x, y = x[pathx], y[pathy]
                X_aligned[idx][:len(x)] = x
                Y_aligned[idx][:len(y)] = y
                if self.verbose > 0:
                    print("{}, distance: {}".format(idx, dist))

            # Fit
            gmm = GaussianMixture(
                n_components=32, covariance_type="full", max_iter=100)
            XY = np.concatenate((X_aligned, Y_aligned),
                                axis=-1).reshape(-1, X.shape[-1] * 2)
            gmm.fit(XY)
            windows = [(0, 0, np.array([1.0]))]  # no delta
            paramgen = MLPG(gmm, windows=windows)
            for idx in range(len(Xc)):
                x = trim_zeros_frames(Xc[idx])
                Xc[idx][:len(x)] = paramgen.transform(x)

        # Finally we can get aligned X
        for idx in range(len(X_aligned)):
            x = X[idx][refined_paths[idx]]
            X_aligned[idx][:len(x)] = x

        return X_aligned, Y_aligned
