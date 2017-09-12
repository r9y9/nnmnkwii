from __future__ import division, print_function, absolute_import

from nnmnkwii.preprocessing import trim_zeros_frames
from nnmnkwii.baseline.gmm import MLPG

from fastdtw import fastdtw

import numpy as np
from numpy.linalg import norm

from sklearn.mixture import GaussianMixture


class DTWAligner(object):
    """Align feature matrices using fastdtw_.

    .. _fastdtw: https://github.com/slaypni/fastdtw

    Attributes:
        dist (function): Distance function. Default is :func:`numpy.linalg.norm`.
        radius (int): Radius parameter in fastdtw_.
        verbose (int): Verbose flag. Default is 0.

    Examples:
        >>> from nnmnkwii.util import example_file_data_sources_for_duration_model
        >>> from nnmnkwii.datasets import FileSourceDataset
        >>> from nnmnkwii.preprocessing.alignment import DTWAligner
        >>> _, X = example_file_data_sources_for_duration_model()
        >>> X = FileSourceDataset(X).asarray()
        >>> X.shape
        (3, 40, 5)
        >>> Y = X.copy()
        >>> X_aligned, Y_aligned = DTWAligner().transform((X, Y))
        >>> X_aligned.shape
        (3, 40, 5)
        >>> Y_aligned.shape
        (3, 40, 5)
    """

    def __init__(self, dist=lambda x, y: norm(x - y), radius=1, verbose=0):
        self.verbose = verbose
        self.dist = dist
        self.radius = radius

    def transform(self, XY):
        X, Y = XY
        assert X.ndim == 3 and Y.ndim == 3

        longer_features = X if X.shape[1] > Y.shape[1] else Y

        X_aligned = np.zeros_like(longer_features)
        Y_aligned = np.zeros_like(longer_features)
        for idx, (x, y) in enumerate(zip(X, Y)):
            x, y = trim_zeros_frames(x), trim_zeros_frames(y)
            dist, path = fastdtw(x, y, radius=self.radius, dist=self.dist)
            dist /= (len(x) + len(y))
            pathx = list(map(lambda l: l[0], path))
            pathy = list(map(lambda l: l[1], path))
            x, y = x[pathx], y[pathy]
            max_len = max(len(x), len(y))
            if max_len > X_aligned.shape[1] or max_len > Y_aligned.shape[1]:
                pad_size = max(max_len - X_aligned.shape[1],
                               max_len > Y_aligned.shape[1])
                X_aligned = np.pad(
                    X_aligned, [(0, 0), (0, pad_size), (0, 0)],
                    mode="constant", constant_values=0)
                Y_aligned = np.pad(
                    Y_aligned, [(0, 0), (0, pad_size), (0, 0)],
                    mode="constant", constant_values=0)
            X_aligned[idx][:len(x)] = x
            Y_aligned[idx][:len(y)] = y
            if self.verbose > 0:
                print("{}, distance: {}".format(idx, dist))
        return X_aligned, Y_aligned


class IterativeDTWAligner(object):
    """Align feature matrices iteratively using GMM-based feature conversion.

    .. _fastdtw: https://github.com/slaypni/fastdtw

    Attributes:
        n_iter (int): Number of iterations.
        dist (function): Distance function
        radius (int): Radius parameter in fastdtw_.
        verbose (int): Verbose flag. Default is 0.
        max_iter_gmm (int): Maximum iteration to train GMM.
        n_components_gmm (int): Number of mixture components in GMM.

    Examples:
        >>> from nnmnkwii.util import example_file_data_sources_for_duration_model
        >>> from nnmnkwii.datasets import FileSourceDataset
        >>> from nnmnkwii.preprocessing.alignment import IterativeDTWAligner
        >>> _, X = example_file_data_sources_for_duration_model()
        >>> X = FileSourceDataset(X).asarray()
        >>> X.shape
        (3, 40, 5)
        >>> Y = X.copy()
        >>> X_aligned, Y_aligned = IterativeDTWAligner(n_iter=1).transform((X, Y))
        >>> X_aligned.shape
        (3, 40, 5)
        >>> Y_aligned.shape
        (3, 40, 5)
    """

    def __init__(self, n_iter=3, dist=lambda x, y: norm(x - y),
                 radius=1, max_iter_gmm=100, n_components_gmm=16, verbose=0):
        self.n_iter = n_iter
        self.dist = dist
        self.radius = radius
        self.max_iter_gmm = max_iter_gmm
        self.n_components_gmm = n_components_gmm
        self.verbose = verbose

    def transform(self, XY):
        X, Y = XY
        assert X.ndim == 3 and Y.ndim == 3

        longer_features = X if X.shape[1] > Y.shape[1] else Y

        Xc = X.copy()  # this will be updated iteratively
        X_aligned = np.zeros_like(longer_features)
        Y_aligned = np.zeros_like(longer_features)
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

                max_len = max(len(x), len(y))
                if max_len > X_aligned.shape[1] or max_len > Y_aligned.shape[1]:
                    pad_size = max(
                        max_len - X_aligned.shape[1],
                        max_len > Y_aligned.shape[1])
                    X_aligned = np.pad(
                        X_aligned, [(0, 0), (0, pad_size), (0, 0)],
                        mode="constant", constant_values=0)
                    Y_aligned = np.pad(
                        Y_aligned, [(0, 0), (0, pad_size), (0, 0)],
                        mode="constant", constant_values=0)

                X_aligned[idx][:len(x)] = x
                Y_aligned[idx][:len(y)] = y
                if self.verbose > 0:
                    print("{}, distance: {}".format(idx, dist))

            # Fit
            gmm = GaussianMixture(
                n_components=self.n_components_gmm,
                covariance_type="full", max_iter=self.max_iter_gmm)
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
