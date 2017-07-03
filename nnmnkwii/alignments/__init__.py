from __future__ import division, print_function, absolute_import

from nnmnkwii.core import Aligner
from nnmnkwii.metrics import melcd
from nnmnkwii.paramgen.gmm import MLStaticParamGen

from fastdtw import fastdtw

import numpy as np

import sklearn.mixture


class DTWAligner(Aligner):
    def __init__(self, verbose=False):
        self.verbose = verbose

    def transform(self, XY):
        X, Y = XY

        aligned_X = np.empty(len(X), dtype=np.object)
        aligned_Y = np.empty(len(X), dtype=np.object)
        for idx, (x, y) in enumerate(zip(X, Y)):
            dist, path = fastdtw(x, y, dist=melcd)
            dist /= (len(x) + len(y))
            pathx = list(map(lambda l: l[0], path))
            pathy = list(map(lambda l: l[1], path))
            aligned_X[idx] = x[pathx]
            aligned_Y[idx] = y[pathy]
            if self.verbose:
                print("{}, distance: {}".format(idx, dist))
        return aligned_X, aligned_Y


class IterativeDTWAligner(Aligner):
    def __init__(self, n_iter=3, verbose=False):
        self.n_iter = n_iter
        self.verbose = verbose

    def transform(self, XY):
        X, Y = XY

        Xc = X.copy()  # this will be updated iteratively
        aligned_X = np.empty(len(X), dtype=np.object)
        aligned_Y = np.empty(len(X), dtype=np.object)
        refined_paths = np.empty(len(X), dtype=np.object)

        for idx in range(self.n_iter):
            for idx, (x, y) in enumerate(zip(Xc, Y)):
                dist, path = fastdtw(x, y, dist=melcd)
                dist /= (len(x) + len(y))
                pathx = list(map(lambda l: l[0], path))
                pathy = list(map(lambda l: l[1], path))

                refined_paths[idx] = pathx
                aligned_X[idx] = x[pathx]
                aligned_Y[idx] = y[pathy]
                if self.verbose:
                    print("{}, distance: {}".format(idx, dist))

            # Fit
            gmm = sklearn.mixture.GaussianMixture(
                n_components=32, covariance_type="full", max_iter=100)
            XY = Joint().transform((aligned_X, aligned_Y))
            gmm.fit(XY)
            paramgen = MLStaticParamGen(gmm)

            for idx in range(len(Xc)):
                Xc[idx] = np.apply_along_axis(paramgen.transform, 1, Xc[idx])

        # Finally we can get aligned X
        for idx in range(len(aligned_X)):
            aligned_X[idx] = X[idx][refined_paths[idx]]

        return aligned_X, aligned_Y


class Joint(Aligner):
    def transform(self, XY):
        Z = None
        X, Y = XY
        for x, y in zip(X, Y):
            xy = np.hstack((x, y))
            if Z is None:
                Z = xy
            else:
                Z = np.vstack((Z, xy))

        return Z
