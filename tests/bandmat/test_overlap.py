"""Tests for functions to do with overlapping subtensors."""

# Copyright 2013, 2014, 2015, 2016, 2017, 2018 Matt Shannon

# This file is part of bandmat.
# See `License` for details of license and warranty.

import random
import unittest

import numpy as np
from nnmnkwii.paramgen import _bandmat as bm
from nnmnkwii.paramgen._bandmat import overlap as bmo
from nnmnkwii.paramgen._bandmat.testhelp import assert_allclose, assert_allequal
from numpy.random import randint, randn
from test_core import gen_BandMat

cc = bm.band_e_bm_common


def rand_bool():
    return randint(0, 2) == 0


def chunk_randomly(xs):
    size = len(xs)

    num_divs = random.choice([0, randint(size // 2 + 1), randint(size + 3)])
    divs = [0] + sorted([randint(size + 1) for _ in range(num_divs)]) + [size]

    for start, end in zip(divs, divs[1:]):
        yield start, end, xs[start:end]


class TestOverlap(unittest.TestCase):
    def test_sum_overlapping_v(self, its=50):
        for it in range(its):
            num_contribs = random.choice([0, 1, randint(10), randint(100)])
            step = random.choice([1, randint(2), randint(10)])
            width = step + random.choice([0, 1, randint(10)])
            overlap = width - step
            vec_size = num_contribs * step + overlap

            contribs = randn(num_contribs, width)
            target = randn(vec_size)
            target_orig = target.copy()

            vec = bmo.sum_overlapping_v(contribs, step=step)
            assert vec.shape == (vec_size,)

            # check target-based version adds to target correctly
            bmo.sum_overlapping_v(contribs, step=step, target=target)
            assert_allclose(target, target_orig + vec)

            if num_contribs == 0:
                # check action for no contributions
                assert_allequal(vec, np.zeros((overlap,)))
            elif num_contribs == 1:
                # check action for a single contribution
                assert_allequal(vec, contribs[0])
            else:
                # check action under splitting list of contributions in two
                split_pos = randint(num_contribs + 1)
                vec_again = np.zeros((vec_size,))
                bmo.sum_overlapping_v(
                    contribs[:split_pos],
                    step=step,
                    target=vec_again[0 : (split_pos * step + overlap)],
                )
                bmo.sum_overlapping_v(
                    contribs[split_pos:],
                    step=step,
                    target=vec_again[(split_pos * step) : vec_size],
                )
                assert_allclose(vec, vec_again)

    def test_sum_overlapping_m(self, its=50):
        for it in range(its):
            num_contribs = random.choice([0, 1, randint(10), randint(100)])
            step = random.choice([1, randint(2), randint(10)])
            width = max(step, 1) + random.choice([0, 1, randint(10)])
            depth = width - 1
            assert depth >= 0
            overlap = width - step
            mat_size = num_contribs * step + overlap

            contribs = randn(num_contribs, width, width)
            target_bm = gen_BandMat(mat_size, l=depth, u=depth)
            target_bm_orig = target_bm.copy()

            mat_bm = bmo.sum_overlapping_m(contribs, step=step)
            assert mat_bm.size == mat_size
            assert mat_bm.l == mat_bm.u == depth

            # check target-based version adds to target_bm correctly
            bmo.sum_overlapping_m(contribs, step=step, target_bm=target_bm)
            assert_allclose(*cc(target_bm, target_bm_orig + mat_bm))

            if num_contribs == 0:
                # check action for no contributions
                assert_allequal(mat_bm.full(), np.zeros((overlap, overlap)))
            elif num_contribs == 1:
                # check action for a single contribution
                assert_allequal(mat_bm.full(), contribs[0])
            else:
                # check action under splitting list of contributions in two
                split_pos = randint(num_contribs + 1)
                mat_bm_again = bm.zeros(depth, depth, mat_size)
                bmo.sum_overlapping_m(
                    contribs[:split_pos],
                    step=step,
                    target_bm=mat_bm_again.sub_matrix_view(
                        0, split_pos * step + overlap
                    ),
                )
                bmo.sum_overlapping_m(
                    contribs[split_pos:],
                    step=step,
                    target_bm=mat_bm_again.sub_matrix_view(split_pos * step, mat_size),
                )
                assert_allclose(*cc(mat_bm, mat_bm_again))

    def test_extract_overlapping_v(self, its=50):
        for it in range(its):
            num_subs = random.choice([0, 1, randint(10), randint(100)])
            step = random.choice([1, randint(1, 10)])
            width = step + random.choice([0, 1, randint(10)])
            overlap = width - step
            vec_size = num_subs * step + overlap

            vec = randn(vec_size)
            target = None if rand_bool() else randn(num_subs, width)

            if target is None:
                subvectors = bmo.extract_overlapping_v(vec, width, step=step)
                assert subvectors.shape == (num_subs, width)
            else:
                bmo.extract_overlapping_v(vec, width, step=step, target=target)
                subvectors = target

            for index in range(num_subs):
                assert_allequal(
                    subvectors[index], vec[(index * step) : (index * step + width)]
                )

    def test_extract_overlapping_m(self, its=50):
        for it in range(its):
            num_subs = random.choice([0, 1, randint(10), randint(100)])
            step = random.choice([1, randint(1, 10)])
            width = step + random.choice([0, 1, randint(10)])
            depth = width - 1
            assert depth >= 0
            overlap = width - step
            mat_size = num_subs * step + overlap

            mat_bm = gen_BandMat(mat_size, l=depth, u=depth)
            target = None if rand_bool() else randn(num_subs, width, width)

            if target is None:
                submats = bmo.extract_overlapping_m(mat_bm, step=step)
                assert submats.shape == (num_subs, width, width)
            else:
                bmo.extract_overlapping_m(mat_bm, step=step, target=target)
                submats = target

            for index in range(num_subs):
                assert_allequal(
                    submats[index],
                    mat_bm.sub_matrix_view(index * step, index * step + width).full(),
                )

    def test_sum_overlapping_v_chunked(self, its=50):
        for it in range(its):
            num_contribs = random.choice([0, 1, randint(10), randint(100)])
            step = random.choice([1, randint(2), randint(10)])
            width = step + random.choice([0, 1, randint(10)])
            overlap = width - step
            vec_size = num_contribs * step + overlap

            contribs = randn(num_contribs, width)
            contribs_chunks = chunk_randomly(contribs)
            target = randn(vec_size)
            target_orig = target.copy()

            bmo.sum_overlapping_v_chunked(contribs_chunks, width, target, step=step)
            vec_good = bmo.sum_overlapping_v(contribs, step=step)
            assert_allclose(target, target_orig + vec_good)

    def test_sum_overlapping_m_chunked(self, its=50):
        for it in range(its):
            num_contribs = random.choice([0, 1, randint(10), randint(100)])
            step = random.choice([1, randint(2), randint(10)])
            width = max(step, 1) + random.choice([0, 1, randint(10)])
            depth = width - 1
            assert depth >= 0
            overlap = width - step
            mat_size = num_contribs * step + overlap

            contribs = randn(num_contribs, width, width)
            contribs_chunks = chunk_randomly(contribs)
            target_bm = gen_BandMat(mat_size, l=depth, u=depth)
            target_bm_orig = target_bm.copy()

            bmo.sum_overlapping_m_chunked(contribs_chunks, target_bm, step=step)
            mat_bm_good = bmo.sum_overlapping_m(contribs, step=step)
            assert_allclose(*cc(target_bm, target_bm_orig + mat_bm_good))

    def test_extract_overlapping_v_chunked(self, its=50):
        for it in range(its):
            num_subs = random.choice([0, 1, randint(10), randint(100)])
            step = random.choice([1, randint(1, 10)])
            width = step + random.choice([0, 1, randint(10)])
            overlap = width - step
            vec_size = num_subs * step + overlap
            chunk_size = random.choice([1, randint(1, 10), randint(1, 10)])

            vec = randn(vec_size)

            indices_remaining = set(range(num_subs))
            subvectors_all = np.empty((num_subs, width))
            for start, end, subvectors in bmo.extract_overlapping_v_chunked(
                vec, width, chunk_size, step=step
            ):
                assert end >= start + 1
                for index in range(start, end):
                    assert index in indices_remaining
                    indices_remaining.remove(index)
                subvectors_all[start:end] = subvectors

            subvectors_good = bmo.extract_overlapping_v(vec, width, step=step)
            assert_allclose(subvectors_all, subvectors_good)

    def test_extract_overlapping_m_chunked(self, its=50):
        for it in range(its):
            num_subs = random.choice([0, 1, randint(10), randint(100)])
            step = random.choice([1, randint(1, 10)])
            width = step + random.choice([0, 1, randint(10)])
            depth = width - 1
            assert depth >= 0
            overlap = width - step
            mat_size = num_subs * step + overlap
            chunk_size = random.choice([1, randint(1, 10), randint(1, 10)])

            mat_bm = gen_BandMat(mat_size, l=depth, u=depth)

            indices_remaining = set(range(num_subs))
            submats_all = np.empty((num_subs, width, width))
            for start, end, submats in bmo.extract_overlapping_m_chunked(
                mat_bm, chunk_size, step=step
            ):
                assert end >= start + 1
                for index in range(start, end):
                    assert index in indices_remaining
                    indices_remaining.remove(index)
                submats_all[start:end] = submats

            submats_good = bmo.extract_overlapping_m(mat_bm, step=step)
            assert_allclose(submats_all, submats_good)


if __name__ == "__main__":
    unittest.main()
