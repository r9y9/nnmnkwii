"""Tests for helper functions for testing."""

# Copyright 2013, 2014, 2015, 2016, 2017, 2018 Matt Shannon

# This file is part of bandmat.
# See `License` for details of license and warranty.

import doctest
import random
import unittest

import numpy as np
from nnmnkwii.paramgen import _bandmat as bm
from nnmnkwii.paramgen._bandmat import full as fl
from nnmnkwii.paramgen._bandmat import testhelp as th
from numpy.random import randint, randn


def rand_bool():
    return randint(0, 2) == 0


def gen_array(ranks=[0, 1, 2, 3]):
    rank = random.choice(ranks)
    shape = tuple([randint(5) for _ in range(rank)])
    return np.asarray(randn(*shape))


def gen_BandMat_simple(size):
    """Generates a random BandMat."""
    l = random.choice([0, 1, randint(0, 10)])
    u = random.choice([0, 1, randint(0, 10)])
    data = randn(l + u + 1, size)
    transposed = rand_bool()
    return bm.BandMat(l, u, data, transposed=transposed)


def load_tests(loader, tests, ignore):
    tests.addTests(doctest.DocTestSuite(th))
    return tests


class TestTestHelp(unittest.TestCase):
    def test_assert_allclose(self):
        a0 = np.array([2.0, 3.0, 4.0])
        a1 = np.array([2.0, 3.0, 4.0])
        a2 = np.array([2.0, 3.0])
        a3 = np.array([2.0, 3.0, 5.0])
        a4 = np.array([[2.0, 3.0, 4.0]])
        th.assert_allclose(a0, a0)
        th.assert_allclose(a0, a1)
        self.assertRaises(AssertionError, th.assert_allclose, a0, a2)
        self.assertRaises(AssertionError, th.assert_allclose, a0, a3)
        self.assertRaises(AssertionError, th.assert_allclose, a0, a4)

    def test_assert_allequal(self):
        a0 = np.array([2.0, 3.0, 4.0])
        a1 = np.array([2.0, 3.0, 4.0])
        a2 = np.array([2.0, 3.0])
        a3 = np.array([2.0, 3.0, 5.0])
        a4 = np.array([[2.0, 3.0, 4.0]])
        th.assert_allequal(a0, a0)
        th.assert_allequal(a0, a1)
        self.assertRaises(AssertionError, th.assert_allequal, a0, a2)
        self.assertRaises(AssertionError, th.assert_allequal, a0, a3)
        self.assertRaises(AssertionError, th.assert_allequal, a0, a4)

    def test_randomize_extra_entries(self, its=50):
        for it in range(its):
            size = random.choice([0, 1, randint(0, 10), randint(0, 100)])
            l = random.choice([0, 1, randint(0, 10)])
            u = random.choice([0, 1, randint(0, 10)])

            mat_rect = randn(l + u + 1, size)
            assert np.all(mat_rect != 0.0)
            fl.zero_extra_entries(l, u, mat_rect)
            th.randomize_extra_entries(l, u, mat_rect)
            assert np.all(mat_rect != 0.0)

            mat_rect = np.zeros((l + u + 1, size))
            assert np.all(mat_rect == 0.0)
            th.randomize_extra_entries(l, u, mat_rect)
            fl.zero_extra_entries(l, u, mat_rect)
            assert np.all(mat_rect == 0.0)

    def test_randomize_extra_entries_bm(self, its=50):
        for it in range(its):
            size = random.choice([0, 1, randint(0, 10), randint(0, 100)])
            mat_bm = gen_BandMat_simple(size)

            mat_full = mat_bm.full()
            th.randomize_extra_entries_bm(mat_bm)
            th.assert_allequal(mat_bm.full(), mat_full)

    def test_get_array_mem(self, its=50):
        # (FIXME : these are not great tests, since not familiar enough with
        #   numpy internals to know what sorts of changes in memory layout are
        #   possible and how they might arise in a realistic program)
        for it in range(its):
            x = gen_array()
            array_mem = th.get_array_mem(x)
            x *= 2.0
            assert th.get_array_mem(x) == array_mem

            x = gen_array()
            array_mem = th.get_array_mem(x)
            x.shape = x.shape + (1,)
            assert th.get_array_mem(x) != array_mem

            x = gen_array(ranks=[1, 2, 3])
            array_mem = th.get_array_mem(x)
            shape = x.shape
            strides = x.strides
            shape_new = x.T.shape
            strides_new = x.T.strides
            if np.prod(shape_new) != 0:
                x.shape = shape_new
                x.strides = strides_new
                if shape_new != shape or strides_new != strides:
                    # FIXME : re-enable once I understand better when this may
                    #   fail (i.e. when memory may be unexpectedly shared).
                    # assert th.get_array_mem(x) != array_mem
                    pass


if __name__ == "__main__":
    unittest.main()
