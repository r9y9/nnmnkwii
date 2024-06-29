"""Tests for linear algebra operations for banded matrices."""

# Copyright 2013, 2014, 2015, 2016, 2017, 2018 Matt Shannon

# This file is part of bandmat.
# See `License` for details of license and warranty.

import random
import re
import unittest

import numpy as np
import numpy.linalg as la
import scipy.linalg as sla
from nnmnkwii.paramgen import _bandmat as bm
from nnmnkwii.paramgen._bandmat import full as fl
from nnmnkwii.paramgen._bandmat import linalg as bla
from nnmnkwii.paramgen._bandmat.testhelp import (
    assert_allclose,
    randomize_extra_entries_bm,
)
from numpy.random import randint, randn
from test_core import gen_BandMat


def rand_bool():
    return randint(0, 2) == 0


def gen_symmetric_BandMat(size, depth=None, transposed=None):
    if depth is None:
        depth = random.choice([0, 1, randint(0, 10)])
    if transposed is None:
        transposed = rand_bool()
    a_bm = gen_BandMat(size, l=depth, u=depth, transposed=transposed)
    b_bm = a_bm + a_bm.T
    randomize_extra_entries_bm(b_bm)
    return b_bm


def gen_pos_def_BandMat(size, depth=None, contrib_rank=2, transposed=None):
    """Generates a random positive definite BandMat."""
    assert contrib_rank >= 0
    if depth is None:
        depth = random.choice([0, 1, randint(0, 10)])
    if transposed is None:
        transposed = rand_bool()
    mat_bm = bm.zeros(depth, depth, size)
    for _ in range(contrib_rank):
        diff = randint(0, depth + 1)
        chol_bm = gen_BandMat(size, l=depth - diff, u=diff)
        bm.dot_mm_plus_equals(chol_bm, chol_bm.T, mat_bm)
    if transposed:
        mat_bm = mat_bm.T
    randomize_extra_entries_bm(mat_bm)
    return mat_bm


def gen_chol_factor_BandMat(size, depth=None, contrib_rank=2, transposed=None):
    """Generates a random Cholesky factor BandMat.

    This works by generating a random positive definite matrix and then
    computing its Cholesky factor, since using a random matrix as a Cholesky
    factor seems to often lead to ill-conditioned matrices.
    """
    if transposed is None:
        transposed = rand_bool()
    mat_bm = gen_pos_def_BandMat(size, depth=depth, contrib_rank=contrib_rank)
    chol_bm = bla.cholesky(mat_bm, lower=rand_bool())
    if transposed:
        chol_bm = chol_bm.T
    assert chol_bm.l == 0 or chol_bm.u == 0
    assert chol_bm.l + chol_bm.u == mat_bm.l
    randomize_extra_entries_bm(chol_bm)
    return chol_bm


class TestLinAlg(unittest.TestCase):
    def test_cholesky_banded_upper_scipy_test(self):
        """Basic test copied from scipy.linalg.tests.test_decomp_cholesky."""
        # Symmetric positive definite banded matrix `a`
        a = np.array(
            [
                [4.0, 1.0, 0.0, 0.0],
                [1.0, 4.0, 0.5, 0.0],
                [0.0, 0.5, 4.0, 0.2],
                [0.0, 0.0, 0.2, 4.0],
            ]
        )
        # Banded storage form of `a`.
        ab = np.array([[-1.0, 1.0, 0.5, 0.2], [4.0, 4.0, 4.0, 4.0]])
        c = bla._cholesky_banded(ab, lower=False)
        ufac = np.zeros_like(a)
        ufac[range(4), range(4)] = c[-1]
        ufac[(0, 1, 2), (1, 2, 3)] = c[0, 1:]
        assert_allclose(a, np.dot(ufac.T, ufac))

    def test_cholesky_banded_lower_scipy_test(self):
        """Basic test copied from scipy.linalg.tests.test_decomp_cholesky."""
        # Symmetric positive definite banded matrix `a`
        a = np.array(
            [
                [4.0, 1.0, 0.0, 0.0],
                [1.0, 4.0, 0.5, 0.0],
                [0.0, 0.5, 4.0, 0.2],
                [0.0, 0.0, 0.2, 4.0],
            ]
        )
        # Banded storage form of `a`.
        ab = np.array([[4.0, 4.0, 4.0, 4.0], [1.0, 0.5, 0.2, -1.0]])
        c = bla._cholesky_banded(ab, lower=True)
        lfac = np.zeros_like(a)
        lfac[range(4), range(4)] = c[0]
        lfac[(1, 2, 3), (0, 1, 2)] = c[1, :3]
        assert_allclose(a, np.dot(lfac, lfac.T))

    def test__cholesky_banded(self, its=100):
        for it in range(its):
            size = random.choice([0, 1, randint(0, 10), randint(0, 100)])
            if rand_bool():
                mat_bm = gen_pos_def_BandMat(size, transposed=False)
            else:
                mat_bm = gen_symmetric_BandMat(size, transposed=False)
                # make it a bit more likely to be pos def
                bm.diag(mat_bm)[:] = np.abs(bm.diag(mat_bm)) + 0.1
            depth = mat_bm.l
            lower = rand_bool()
            if lower:
                mat_half_data = mat_bm.data[depth:]
            else:
                mat_half_data = mat_bm.data[: (depth + 1)]
            overwrite = rand_bool()

            mat_half_data_arg = mat_half_data.copy()
            try:
                chol_data = bla._cholesky_banded(
                    mat_half_data_arg, overwrite_ab=overwrite, lower=lower
                )
            except la.LinAlgError as e:
                # First part of the message is e.g. "2-th leading minor".
                msgRe = r"^" + re.escape(str(e)[:15]) + r".*not positive definite$"
                with self.assertRaisesRegexp(la.LinAlgError, msgRe):
                    sla.cholesky(mat_bm.full(), lower=lower)
            else:
                assert np.shape(chol_data) == (depth + 1, size)
                if lower:
                    chol_bm = bm.BandMat(depth, 0, chol_data)
                    mat_bm_again = bm.dot_mm(chol_bm, chol_bm.T)
                else:
                    chol_bm = bm.BandMat(0, depth, chol_data)
                    mat_bm_again = bm.dot_mm(chol_bm.T, chol_bm)
                assert_allclose(mat_bm_again.full(), mat_bm.full())

                if size > 0:
                    self.assertEqual(
                        np.may_share_memory(chol_data, mat_half_data_arg), overwrite
                    )

            if not overwrite:
                assert np.all(mat_half_data_arg == mat_half_data)

    def test__solve_triangular_banded(self, its=100):
        for it in range(its):
            size = random.choice([0, 1, randint(0, 10), randint(0, 100)])
            b = randn(size)
            chol_bm = gen_chol_factor_BandMat(size, transposed=False)
            chol_data = chol_bm.data
            depth = chol_bm.l + chol_bm.u
            lower = chol_bm.u == 0
            if size > 0 and rand_bool() and rand_bool():
                badFrame = randint(size)
                chol_data[0 if lower else depth, badFrame] = 0.0
            else:
                badFrame = None
            transposed = rand_bool()
            overwrite_b = rand_bool()
            chol_full = chol_bm.full()

            b_arg = b.copy()
            if badFrame is not None:
                msg = "singular matrix: resolution failed at diagonal %d" % badFrame
                msgRe = "^" + re.escape(msg) + "$"
                with self.assertRaisesRegexp(la.LinAlgError, msgRe):
                    bla._solve_triangular_banded(
                        chol_data,
                        b_arg,
                        transposed=transposed,
                        lower=lower,
                        overwrite_b=overwrite_b,
                    )
                with self.assertRaisesRegexp(la.LinAlgError, msgRe):
                    sla.solve_triangular(chol_full, b, trans=transposed, lower=lower)
            else:
                x = bla._solve_triangular_banded(
                    chol_data,
                    b_arg,
                    transposed=transposed,
                    lower=lower,
                    overwrite_b=overwrite_b,
                )
                if transposed:
                    assert_allclose(bm.dot_mv(chol_bm.T, x), b)
                else:
                    assert_allclose(bm.dot_mv(chol_bm, x), b)
                if size == 0:
                    x_good = np.zeros((size,))
                else:
                    x_good = sla.solve_triangular(
                        chol_full, b, trans=transposed, lower=lower
                    )
                assert_allclose(x, x_good)
                assert not np.may_share_memory(x, chol_data)
                if size > 0:
                    self.assertEqual(np.may_share_memory(x, b_arg), overwrite_b)

            if not overwrite_b:
                assert np.all(b_arg == b)

    def test_cholesky(self, its=50):
        for it in range(its):
            size = random.choice([0, 1, randint(0, 10), randint(0, 100)])
            mat_bm = gen_pos_def_BandMat(size)
            depth = mat_bm.l
            lower = rand_bool()
            alternative = rand_bool()

            chol_bm = bla.cholesky(mat_bm, lower=lower, alternative=alternative)
            assert chol_bm.l == (depth if lower else 0)
            assert chol_bm.u == (0 if lower else depth)
            assert not np.may_share_memory(chol_bm.data, mat_bm.data)

            if lower != alternative:
                mat_bm_again = bm.dot_mm(chol_bm, chol_bm.T)
            else:
                mat_bm_again = bm.dot_mm(chol_bm.T, chol_bm)
            assert_allclose(mat_bm_again.full(), mat_bm.full())

    def test_solve_triangular(self, its=50):
        for it in range(its):
            size = random.choice([0, 1, randint(0, 10), randint(0, 100)])
            b = randn(size)
            chol_bm = gen_chol_factor_BandMat(size)
            depth = chol_bm.l + chol_bm.u
            lower = chol_bm.u == 0
            chol_lower_bm = chol_bm if lower else chol_bm.T
            chol_full = chol_bm.full()

            x = bla.solve_triangular(chol_bm, b)
            assert_allclose(bm.dot_mv(chol_bm, x), b)
            if size == 0:
                x_good = np.zeros((size,))
            else:
                x_good = sla.solve_triangular(chol_full, b, lower=lower)
            assert_allclose(x, x_good)
            assert not np.may_share_memory(x, chol_bm.data)
            assert not np.may_share_memory(x, b)

    def test_cho_solve(self, its=50):
        for it in range(its):
            size = random.choice([0, 1, randint(0, 10), randint(0, 100)])
            b = randn(size)
            chol_bm = gen_chol_factor_BandMat(size)
            depth = chol_bm.l + chol_bm.u
            lower = chol_bm.u == 0
            chol_lower_bm = chol_bm if lower else chol_bm.T
            chol_full = chol_bm.full()

            x = bla.cho_solve(chol_bm, b)
            assert_allclose(bm.dot_mv(chol_lower_bm, bm.dot_mv(chol_lower_bm.T, x)), b)
            if size == 0:
                x_good = np.zeros((size,))
            else:
                x_good = sla.cho_solve((chol_full, lower), b)
            assert_allclose(x, x_good)
            assert not np.may_share_memory(x, chol_bm.data)
            assert not np.may_share_memory(x, b)

    def test_solve(self, its=50):
        for it in range(its):
            size = random.choice([0, 1, randint(0, 10), randint(0, 100)])
            b = randn(size)
            # the below tries to ensure the matrix is well-conditioned
            a_bm = gen_BandMat(size) + bm.diag(np.ones((size,)) * 10.0)
            a_full = a_bm.full()

            x = bla.solve(a_bm, b)
            assert_allclose(bm.dot_mv(a_bm, x), b)
            if size == 0:
                x_good = np.zeros((size,))
            else:
                x_good = sla.solve(a_full, b)
            assert_allclose(x, x_good)
            assert not np.may_share_memory(x, a_bm.data)
            assert not np.may_share_memory(x, b)

    def test_solveh(self, its=50):
        for it in range(its):
            size = random.choice([0, 1, randint(0, 10), randint(0, 100)])
            b = randn(size)
            a_bm = gen_pos_def_BandMat(size)
            a_full = a_bm.full()

            x = bla.solveh(a_bm, b)
            assert_allclose(bm.dot_mv(a_bm, x), b)
            if size == 0:
                x_good = np.zeros((size,))
            else:
                # x_good = sla.solve(a_full, b, sym_pos=True)
                x_good = sla.solve(a_full, b, assume_a="pos")
            assert_allclose(x, x_good)
            assert not np.may_share_memory(x, a_bm.data)
            assert not np.may_share_memory(x, b)

    def test_band_of_inverse_from_chol(self, its=50):
        for it in range(its):
            size = random.choice([0, 1, randint(0, 10), randint(0, 100)])
            chol_bm = gen_chol_factor_BandMat(size)
            depth = chol_bm.l + chol_bm.u

            band_of_inv_bm = bla.band_of_inverse_from_chol(chol_bm)
            assert not np.may_share_memory(band_of_inv_bm.data, chol_bm.data)

            mat_bm = (
                bm.dot_mm(chol_bm, chol_bm.T)
                if chol_bm.u == 0
                else bm.dot_mm(chol_bm.T, chol_bm)
            )
            band_of_inv_full_good = fl.band_ec(
                depth, depth, np.eye(0, 0) if size == 0 else la.inv(mat_bm.full())
            )
            assert_allclose(band_of_inv_bm.full(), band_of_inv_full_good)

    def test_band_of_inverse(self, its=50):
        for it in range(its):
            size = random.choice([0, 1, randint(0, 10), randint(0, 100)])
            mat_bm = gen_pos_def_BandMat(size)
            depth = mat_bm.l

            band_of_inv_bm = bla.band_of_inverse(mat_bm)
            assert not np.may_share_memory(band_of_inv_bm.data, mat_bm.data)

            band_of_inv_full_good = fl.band_ec(
                depth, depth, np.eye(0, 0) if size == 0 else la.inv(mat_bm.full())
            )
            assert_allclose(band_of_inv_bm.full(), band_of_inv_full_good)


if __name__ == "__main__":
    unittest.main()
