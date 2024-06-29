"""Tests for operations involving the bands of square matrices."""

# Copyright 2013, 2014, 2015, 2016, 2017, 2018 Matt Shannon

# This file is part of bandmat.
# See `License` for details of license and warranty.

import random
import unittest

import numpy as np
from nnmnkwii.paramgen._bandmat import full as fl
from nnmnkwii.paramgen._bandmat.testhelp import (
    assert_allclose,
    assert_allequal,
    get_array_mem,
)
from numpy.random import randint, randn


class TestFull(unittest.TestCase):
    def test_band_c_basis(self, its=100):
        """Checks band_c behaves correctly on canonical basis matrices."""
        for it in range(its):
            l = random.choice([0, 1, randint(0, 10)])
            u = random.choice([0, 1, randint(0, 10)])
            # size >= 1 (there are no canonical basis matrices if size == 0)
            size = random.choice([1, 2, randint(1, 10), randint(1, 100)])

            # pick a random canonical basis matrix
            i = randint(-u, l + 1)
            j = randint(size)
            mat_rect = np.zeros((l + u + 1, size))
            mat_rect[u + i, j] = 1.0

            mat_full = fl.band_c(l, u, mat_rect)

            k = i + j
            mat_full_good = np.zeros((size, size))
            if 0 <= k < size:
                mat_full_good[k, j] = 1.0

            assert_allequal(mat_full, mat_full_good)

    def test_band_c_linear(self, its=100):
        """Checks band_c is linear."""
        for it in range(its):
            l = random.choice([0, 1, randint(0, 10)])
            u = random.choice([0, 1, randint(0, 10)])
            size = random.choice([0, 1, randint(0, 10), randint(0, 100)])

            # check additive
            mat_rect1 = randn(l + u + 1, size)
            mat_rect2 = randn(l + u + 1, size)
            assert_allclose(
                fl.band_c(l, u, mat_rect1 + mat_rect2),
                fl.band_c(l, u, mat_rect1) + fl.band_c(l, u, mat_rect2),
            )

            # check homogeneous
            mat_rect = randn(l + u + 1, size)
            mult = random.choice([0.0, randn(), randn(), randn()])
            assert_allclose(
                fl.band_c(l, u, mat_rect * mult), fl.band_c(l, u, mat_rect) * mult
            )

            # check output is a newly-created array
            mat_full = fl.band_c(l, u, mat_rect)
            assert not np.may_share_memory(mat_full, mat_rect)

    def test_band_e_basis(self, its=100):
        """Checks band_e behaves correctly on canonical basis matrices."""
        for it in range(its):
            l = random.choice([0, 1, randint(0, 10)])
            u = random.choice([0, 1, randint(0, 10)])
            # size >= 1 (there are no canonical basis matrices if size == 0)
            size = random.choice([1, 2, randint(1, 10), randint(1, 100)])

            # pick a random canonical basis matrix
            k = randint(size)
            j = randint(size)
            mat_full = np.zeros((size, size))
            mat_full[k, j] = 1.0

            mat_rect = fl.band_e(l, u, mat_full)

            i = k - j
            mat_rect_good = np.zeros((l + u + 1, size))
            if -u <= i <= l:
                mat_rect_good[u + i, j] = 1.0

            assert_allequal(mat_rect, mat_rect_good)

    def test_band_e_linear(self, its=100):
        """Checks band_e is linear."""
        for it in range(its):
            l = random.choice([0, 1, randint(0, 10)])
            u = random.choice([0, 1, randint(0, 10)])
            size = random.choice([0, 1, randint(0, 10), randint(0, 100)])

            # check additive
            mat_full1 = randn(size, size)
            mat_full2 = randn(size, size)
            assert_allclose(
                fl.band_e(l, u, mat_full1 + mat_full2),
                fl.band_e(l, u, mat_full1) + fl.band_e(l, u, mat_full2),
            )

            # check homogeneous
            mat_full = randn(size, size)
            mult = random.choice([0.0, randn(), randn(), randn()])
            assert_allclose(
                fl.band_e(l, u, mat_full * mult), fl.band_e(l, u, mat_full) * mult
            )

            # check output is a newly-created array
            mat_rect = fl.band_e(l, u, mat_full)
            assert not np.may_share_memory(mat_rect, mat_full)

    def test_zero_extra_entries(self, its=100):
        """Checks zero_extra_entries against its equivalent definition."""
        for it in range(its):
            l = random.choice([0, 1, randint(0, 10)])
            u = random.choice([0, 1, randint(0, 10)])
            size = random.choice([0, 1, randint(0, 10), randint(0, 100)])
            mat_rect = randn(l + u + 1, size)
            mat_rect_good = mat_rect.copy()
            array_mem = get_array_mem(mat_rect)

            fl.zero_extra_entries(l, u, mat_rect)
            mat_rect_good[:] = fl.band_e(l, u, fl.band_c(l, u, mat_rect_good))
            assert_allequal(mat_rect, mat_rect_good)
            assert get_array_mem(mat_rect) == array_mem

    def test_band_ce(self, its=100):
        """Checks band_ce against its definition and required properties."""
        for it in range(its):
            l = random.choice([0, 1, randint(0, 10)])
            u = random.choice([0, 1, randint(0, 10)])
            size = random.choice([0, 1, randint(0, 10), randint(0, 100)])
            mat_rect = randn(l + u + 1, size)

            mat_rect_new = fl.band_ce(l, u, mat_rect)
            mat_rect_new_good = fl.band_e(l, u, fl.band_c(l, u, mat_rect))
            assert_allequal(mat_rect_new, mat_rect_new_good)
            assert not np.may_share_memory(mat_rect_new, mat_rect)

            # check idempotent
            assert_allequal(fl.band_ce(l, u, mat_rect_new), mat_rect_new)

    def test_band_ec(self, its=100):
        """Checks band_ec against its definition and required properties."""
        for it in range(its):
            l = random.choice([0, 1, randint(0, 10)])
            u = random.choice([0, 1, randint(0, 10)])
            size = random.choice([0, 1, randint(0, 10), randint(0, 100)])
            mat_full = randn(size, size)

            mat_full_new = fl.band_ec(l, u, mat_full)
            mat_full_new_good = fl.band_c(l, u, fl.band_e(l, u, mat_full))
            assert_allequal(mat_full_new, mat_full_new_good)
            assert not np.may_share_memory(mat_full_new, mat_full)

            # check idempotent
            assert_allequal(fl.band_ec(l, u, mat_full_new), mat_full_new)

    def test_band_cTe(self, its=100):
        """Checks band_cTe against its definition and required properties."""
        for it in range(its):
            l = random.choice([0, 1, randint(0, 10)])
            u = random.choice([0, 1, randint(0, 10)])
            size = random.choice([0, 1, randint(0, 10), randint(0, 100)])
            mat_rect = randn(l + u + 1, size)

            mat_rect_new = fl.band_cTe(l, u, mat_rect)
            mat_rect_new_good = fl.band_e(u, l, fl.band_c(l, u, mat_rect).T)
            assert_allequal(mat_rect_new, mat_rect_new_good)
            assert not np.may_share_memory(mat_rect_new, mat_rect)

            # check a property to do with doing band_cTe twice
            assert_allequal(fl.band_cTe(u, l, mat_rect_new), fl.band_ce(l, u, mat_rect))

            # check version that uses pre-specified target
            mat_rect_new2 = np.empty((l + u + 1, size))
            array_mem = get_array_mem(mat_rect, mat_rect_new2)
            ret = fl.band_cTe(l, u, mat_rect, target_rect=mat_rect_new2)
            self.assertIsNone(ret)
            assert_allequal(mat_rect_new2, mat_rect_new)
            assert get_array_mem(mat_rect, mat_rect_new2) == array_mem


if __name__ == "__main__":
    unittest.main()
