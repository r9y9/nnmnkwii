#!/usr/bin/python
# coding: utf-8

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.mixture.gaussian_mixture import _compute_precision_cholesky
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg


def split_gmm_means(means):
    D = means.shape[-1] // 2
    return means[:, 0:D], means[:, D:]


def split_gmm_covariances(covariances):
    D = covariances.shape[-1] // 2
    covarXX = covariances[:, :D, :D]
    covarXY = covariances[:, :D, D:]
    covarYX = covariances[:, D:, :D]
    covarYY = covariances[:, D:, D:]
    return covarXX, covarXY, covarYX, covarYY


# TODO: this can be refactored to be more flexible
# e.g. take `swap` and `diff` out of the class
class MLParameterGenerationBase(object):
    """Base class for Maximum likelihood Parameter Generation (MLPG)

    Notation
    --------
    Source speaker's feature: X = {x_t}, 0 <= t < T
    Target speaker's feature: Y = {y_t}, 0 <= t < T
    where T is the number of time frames.

    Parameters
    ----------
    gmm : sklearn.mixture.GaussianMixture
        Gaussian Mixture Models of source and target joint features

    swap : bool
        True: source -> target
        False target -> source

    diff : bool
        Convert GMM -> DIFFGMM if True

    Attributes
    ----------
    num_mixtures : int
        the number of Gaussian mixtures
    weights : array, shape (`num_mixtures`)
        weights for each gaussian
    src_means : array, shape (`num_mixtures`, `order of spectral feature`)
        means of GMM for a source speaker
    tgt_means : array, shape (`num_mixtures`, `order of spectral feature`)
        means of GMM for a target speaker
    covarXX : array, shape (`num_mixtures`, `order of spectral feature`,
        `order of spectral feature`)
        variance matrix of source speaker's spectral feature
    covarXY : array, shape (`num_mixtures`, `order of spectral feature`,
        `order of spectral feature`)
        covariance matrix of source and target speaker's spectral feature
    covarYX : array, shape (`num_mixtures`, `order of spectral feature`,
        `order of spectral feature`)
        covariance matrix of target and source speaker's spectral feature
    covarYY : array, shape (`num_mixtures`, `order of spectral feature`,
        `order of spectral feature`)
        variance matrix of target speaker's spectral feature

    D : array, shape (`num_mixtures`, `order of spectral feature`,
        `order of spectral feature`)
        covariance matrices of target static spectral features
    px : sklearn.mixture.GaussianMixture
        Gaussian Mixture Models of source speaker's features

    Reference
    ---------
      - [Toda 2007] Voice Conversion Based on Maximum Likelihood Estimation
        of Spectral Parameter Trajectory.
        http://isw3.naist.jp/~tomoki/Tomoki/Journals/IEEE-Nov-2007_MLVC.pdf
    """

    def __init__(self, gmm, swap=False, diff=False):
        # D: static + delta dim
        D = gmm.means_.shape[1] // 2
        self.num_mixtures = gmm.means_.shape[0]
        self.weights = gmm.weights_

        # Split source and target parameters from joint GMM
        self.src_means = gmm.means_[:, 0:D]
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
            self.covarYX, self.covarXY = self.XY, self.covarYX

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

        Parameters
        ----------
        src : array, shape (`order of spectral feature`)
            source speaker's spectral feature that will be transformed

        Return
        ------
        converted spectral feature
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


class MLParameterGeneration(MLParameterGenerationBase):
    """Maximum likelihood parameter generation (MLPG)

    Parameters
    ----------
    gmm : scipy.mixture.GMM
        Gaussian Mixture Models of source and target speaker joint features

    static_dim : int
        Dimention of static feature

    swap : bool (default=False)
        True: source -> target
        False target -> source

    diff : bool
        GMM -> DIFFGMM if True

    Attributes
    ----------
    TODO


    Reference
    ---------
      - [Toda 2007] Voice Conversion Based on Maximum Likelihood Estimation
        of Spectral Parameter Trajectory.
        http://isw3.naist.jp/~tomoki/Tomoki/Journals/IEEE-Nov-2007_MLVC.pdf
    """

    def __init__(self, gmm, static_dim, swap=False, diff=False):
        super(MLParameterGeneration, self).__init__(gmm, swap, diff)
        self.static_dim = static_dim
        self.T = -1
        self.D = None
        self.W = None

    def transform(self, src):
        """Mapping source spectral feature x to target spectral feature y
        so that maximize the likelihood of y given x.

        Parameters
        ----------
        src : array, shape (`the number of frames`, `the order of spectral feature`)
            a sequence of source speaker's spectral feature that will be
            transformed

        Returns
        -------
        a sequence of transformed spectral features
        """
        T, feature_dim = src.shape[0], src.shape[1]

        if feature_dim == self.static_dim:
            return super(MLParameterGeneration, self).transform(src)
        # TODO
        assert feature_dim == self.static_dim * 2

        # alias
        D = self.static_dim

        if T != self.T:
            self.T = T
            self.W = construct_weight_matrix(T, D)
            assert self.W.shape == (2 * D * T, D * T)

        # A suboptimum mixture sequence  (eq.37)
        optimum_mix = self.px.predict(src)

        # Compute E eq.(40)
        self.E = np.zeros((T, 2 * D))
        for t in range(T):
            m = optimum_mix[t]  # estimated mixture index at time t
            xx = np.linalg.solve(self.covarXX[m], src[t] - self.src_means[m])
            # Eq. (22)
            self.E[t] = self.tgt_means[m] + np.dot(self.covarYX[m], xx)
        self.E = self.E.flatten()

        # Compute D eq.(41). Note that self.D represents D^-1.
        self.D = np.zeros((T, 2 * D, 2 * D))
        for t in range(T):
            m = optimum_mix[t]
            xx_inv_xy = np.linalg.solve(self.covarXX[m], self.covarXY[m])
            # Eq. (23)
            self.D[t] = self.covarYY[m] - np.dot(self.covarYX[m], xx_inv_xy)
            self.D[t] = np.linalg.inv(self.D[t])
        self.D = scipy.linalg.block_diag(*self.D)

        # represent D as a sparse matrix
        self.D = scipy.sparse.csr_matrix(self.D)

        # Compute target static features
        # eq.(39)
        covar = self.W.T.dot(self.D.dot(self.W))
        y = scipy.sparse.linalg.spsolve(covar, self.W.T.dot(self.D.dot(self.E)),
                                        use_umfpack=False)
        return y.reshape((T, D))


def construct_weight_matrix(T, D):
    # Construct Weight matrix W
    # Eq.(25) ~ (28)

    data = []
    indices = []
    indptr = [0]
    cum = 1
    for t in range(T):
        data.extend(np.ones(D))
        indices.extend(np.arange(t * D, (t + 1) * D))
        indptr.extend(np.arange(cum, cum + D))
        cum += D

        if t == 0:
            data.extend(np.ones(D) * 0.5)
            indices.extend(np.arange((t + 1) * D, (t + 2) * D))
        elif t == T - 1:
            data.extend(np.ones(D) * -0.5)
            indices.extend(np.arange((t - 1) * D, t * D))
        else:
            d = np.empty(2 * D)
            d[0::2] = np.ones(D) * -0.5
            d[1::2] = np.ones(D) * 0.5
            ind = np.empty(2 * D)
            ind[0::2] = np.arange((t - 1) * D, t * D)
            ind[1::2] = np.arange((t + 1) * D, (t + 2) * D)
            data.extend(d)
            indices.extend(ind)

        if t == 0 or t == T - 1:
            indptr.extend(np.arange(cum, cum + D))
            cum += D
        else:
            indptr.extend(np.arange(cum + 1, cum + 1 + D * 2, 2))
            cum += 2 * D

    W = scipy.sparse.csr_matrix(
        (data, indices, indptr), shape=(2 * D * T, D * T))
    return W
