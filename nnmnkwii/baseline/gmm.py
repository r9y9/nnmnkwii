import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.mixture.gaussian_mixture import _compute_precision_cholesky

from nnmnkwii.paramgen import mlpg

# TODO: this can be refactored to be more flexible
# e.g. take `swap` and `diff` out of the class


class MLPGBase(object):
    def __init__(self, gmm, swap=False, diff=False):
        assert gmm.covariance_type == "full"
        # D: static + delta dim
        D = gmm.means_.shape[1] // 2
        self.num_mixtures = gmm.means_.shape[0]
        self.weights = gmm.weights_

        # Split source and target parameters from joint GMM
        self.src_means = gmm.means_[:, :D]
        self.tgt_means = gmm.means_[:, D:]
        self.covarXX = gmm.covariances_[:, :D, :D]
        self.covarXY = gmm.covariances_[:, :D, D:]
        self.covarYX = gmm.covariances_[:, D:, :D]
        self.covarYY = gmm.covariances_[:, D:, D:]

        if diff:
            self.tgt_means = self.tgt_means - self.src_means
            self.covarYY = self.covarXX + self.covarYY - self.covarXY - self.covarYX
            self.covarXY = self.covarXY - self.covarXX
            self.covarYX = self.covarXY.transpose(0, 2, 1)

        # swap src and target parameters
        if swap:
            self.tgt_means, self.src_means = self.src_means, self.tgt_means
            self.covarYY, self.covarXX = self.covarXX, self.covarYY
            self.covarYX, self.covarXY = self.covarXY, self.covarYX

        # p(x), which is used to compute posterior prob. for a given source
        # spectral feature in mapping stage.
        self.px = GaussianMixture(
            n_components=self.num_mixtures, covariance_type="full")
        self.px.means_ = self.src_means
        self.px.covariances_ = self.covarXX
        self.px.weights_ = self.weights
        self.px.precisions_cholesky_ = _compute_precision_cholesky(
            self.px.covariances_, "full")

    def transform(self, src):
        if src.ndim == 2:
            tgt = np.zeros_like(src)
            for idx, x in enumerate(src):
                y = self._transform_frame(x)
                tgt[idx][:len(y)] = y
            return tgt
        else:
            return self._transform_frame(src)

    def _transform_frame(self, src):
        """Mapping source spectral feature x to target spectral feature y
        so that minimize the mean least squared error.
        More specifically, it returns the value E(p(y|x)].

        Args:
            src (array): shape (`order of spectral feature`) source speaker's
                spectral feature that will be transformed

        Returns:
            array: converted spectral feature
        """
        D = len(src)

        # Eq.(11)
        E = np.zeros((self.num_mixtures, D))
        for m in range(self.num_mixtures):
            xx = np.linalg.solve(self.covarXX[m], src - self.src_means[m])
            E[m] = self.tgt_means[m] + self.covarYX[m].dot(xx)

        # Eq.(9) p(m|x)
        posterior = self.px.predict_proba(np.atleast_2d(src))

        # Eq.(13) conditinal mean E[p(y|x)]
        return posterior.dot(E).flatten()


class MLPG(MLPGBase):
    """Maximum likelihood Parameter Generation (MLPG) for GMM-basd voice
    conversion [1]_.

    Notes:
        - Source speaker's feature: ``X = {x_t}, 0 <= t < T``
        - Target speaker's feature: ``Y = {y_t}, 0 <= t < T``

        where T is the number of time frames.

    See papar [1]_ for details.

    The code was adapted from https://gist.github.com/r9y9/88bda659c97f46f42525.

    Args:
        gmm (sklearn.mixture.GaussianMixture): Gaussian Mixture Models of
            source and target joint features.
        windows (list): List of windows. See :func:`nnmnkwii.functions.mlpg` for
            details.
        swap (bool): If True, source -> target, otherwise target -> source.
        diff (bool): Convert GMM -> DIFFGMM if True.

    Attributes:
        num_mixtures (int): The number of Gaussian mixtures
        weights (array):  shape (`num_mixtures`), weights for each gaussian
        src_means (array): shape (`num_mixtures`, `order of spectral feature`)
            means of GMM for a source speaker
        tgt_means (array): shape (`num_mixtures`, `order of spectral feature`)
            means of GMM for a target speaker
        covarXX (array): shape (`num_mixtures`, `order of spectral feature`,
            `order of spectral feature`)
            variance matrix of source speaker's spectral feature
        covarXY (array): shape (`num_mixtures`, `order of spectral feature`,
            `order of spectral feature`)
            covariance matrix of source and target speaker's spectral feature
        covarYX (array): shape (`num_mixtures`, `order of spectral feature`,
            `order of spectral feature`)
            covariance matrix of target and source speaker's spectral feature
        covarYY (array): shape (`num_mixtures`, `order of spectral feature`,
            `order of spectral feature`) variance matrix of target speaker's
            spectral feature
        D (array): shape (`num_mixtures`, `order of spectral feature`,
            `order of spectral feature`) covariance matrices of target static
            spectral features
        px (sklearn.mixture.GaussianMixture): Gaussian Mixture Models of source
            speaker's features

    Examples:
        >>> from sklearn.mixture import GaussianMixture
        >>> from nnmnkwii.baseline.gmm import MLPG
        >>> import numpy as np
        >>> static_dim, T = 24, 10
        >>> windows = [
        ...     (0, 0, np.array([1.0])),
        ...     (1, 1, np.array([-0.5, 0.0, 0.5])),
        ...     (1, 1, np.array([1.0, -2.0, 1.0])),
        ... ]
        >>> src = np.random.rand(T, static_dim * len(windows))
        >>> tgt = np.random.rand(T, static_dim * len(windows))
        >>> XY = np.concatenate((src, tgt), axis=-1) # pseudo parallel data
        >>> gmm = GaussianMixture(n_components=4)
        >>> _ = gmm.fit(XY)
        >>> paramgen = MLPG(gmm, windows=windows)
        >>> generated = paramgen.transform(src)
        >>> assert generated.shape == (T, static_dim)

    See also:
        :class:`nnmnkwii.preprocessing.alignment.IterativeDTWAligner`.

    .. [1] [Toda 2007] Voice Conversion Based on Maximum Likelihood Estimation
      of Spectral Parameter Trajectory.
    """

    def __init__(self, gmm, windows=None, swap=False, diff=False):
        super(MLPG, self).__init__(gmm, swap, diff)
        if windows is None:
            windows = [
                (0, 0, np.array([1.0])),
                (1, 1, np.array([-0.5, 0.0, 0.5])),
            ]
        self.windows = windows
        self.static_dim = gmm.means_.shape[-1] // 2 // len(windows)

    def transform(self, src):
        """Mapping source feature x to target feature y so that maximize the
        likelihood of y given x.

        Args:
            src (array): shape (`the number of frames`, `the order of spectral
                feature`) a sequence of source speaker's spectral feature that
                will be transformed.

        Returns:
            array: a sequence of transformed features
        """
        T, feature_dim = src.shape[0], src.shape[1]

        if feature_dim == self.static_dim:
            return super(MLPG, self).transform(src)

        # A suboptimum mixture sequence  (eq.37)
        optimum_mix = self.px.predict(src)

        # Compute E eq.(40)
        E = np.empty((T, feature_dim))
        for t in range(T):
            m = optimum_mix[t]  # estimated mixture index at time t
            xx = np.linalg.solve(self.covarXX[m], src[t] - self.src_means[m])
            # Eq. (22)
            E[t] = self.tgt_means[m] + np.dot(self.covarYX[m], xx)

        # Compute D eq.(23)
        # Approximated variances with diagonals so that we can do MLPG
        # efficiently in dimention-wise manner
        D = np.empty((T, feature_dim))
        for t in range(T):
            m = optimum_mix[t]
            # Eq. (23), with approximating covariances as diagonals
            D[t] = np.diag(self.covarYY[m]) - np.diag(self.covarYX[m]) / \
                np.diag(self.covarXX[m]) * np.diag(self.covarXY[m])

        # Once we have mean and variance over frames, then we can do MLPG
        return mlpg(E, D, self.windows)
