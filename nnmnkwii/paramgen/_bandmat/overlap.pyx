# cython: language_level=3

"""Functions to do with overlapping subtensors."""

# Copyright 2013, 2014, 2015, 2016, 2017, 2018 Matt Shannon

# This file is part of bandmat.
# See `License` for details of license and warranty.

from nnmnkwii.paramgen import _bandmat as bm

import numpy as np

cimport numpy as cnp
cimport cython

cnp.import_array()
cnp.import_ufunc()

@cython.boundscheck(False)
def sum_overlapping_v(cnp.ndarray[cnp.float64_t, ndim=2] contribs,
                      unsigned long step=1,
                      cnp.ndarray[cnp.float64_t, ndim=1] target=None):
    """Computes the overlapped sum of a sequence of vectors.

    The overlapped sum of a sequence of vectors is defined as follows.
    Suppose the vectors in `contribs` are "laid out" along some larger vector
    such that each element of `contribs` is offset by `step` indices relative
    to the previous element.
    For example `contribs[0]` occupies the left edge of the larger vector,
    `contribs[1]` starts at index `step` and `contribs[2]` starts at index
    `step * 2`.
    The overlapped sum is the sum of these vectors laid out in this way, which
    is a vector.

    If `target` is None then a new vector is returned; otherwise the overlapped
    sum is added to the vector `target`.
    If `contribs` has shape `(num_contribs, width)` then the returned vector
    (or `target` if specified) has size `num_contribs * step + width - step`.
    The value of `width` here must be at least `step`.
    """
    assert contribs.shape[1] >= step

    cdef unsigned long width = contribs.shape[1]
    cdef unsigned long num_contribs = contribs.shape[0]
    cdef unsigned long vec_size = num_contribs * step + width - step
    cdef cnp.ndarray[cnp.float64_t, ndim=1] vec

    if target is None:
        vec = np.zeros((vec_size,), dtype=contribs.dtype)
    else:
        assert target.shape[0] == vec_size
        vec = target

    cdef unsigned long index, frame, k

    for index in range(num_contribs):
        frame = index * step
        for k in range(width):
            vec[frame + k] += contribs[index, k]

    if target is None:
        return vec
    else:
        return

@cython.boundscheck(False)
def sum_overlapping_m(cnp.ndarray[cnp.float64_t, ndim=3] contribs,
                      unsigned long step=1,
                      target_bm=None):
    """Computes the overlapped sum of a sequence of square matrices.

    The overlapped sum of a sequence of matrices is defined as follows.
    Suppose the matrices in `contribs` are "laid out" along the diagonal of
    some larger matrix such that each element of `contribs` is `step` indices
    further down and `step` indices further right than the previous element.
    For example `contribs[0]` occupies the top left corner of the larger matrix
    and `contribs[1]` is `step` indices down and right of this.
    The overlapped sum is the sum of these matrices laid out in this way, which
    is a banded matrix.
    For this function the contributions are square, so the resulting banded
    matrix is also square.

    If `target_bm` is None then a new BandMat is returned; otherwise the
    overlapped sum is added to the BandMat `target_bm`.
    If `contribs` has shape `(num_contribs, width, width)` then the returned
    BandMat (or `target_bm` if specified) has upper and lower bandwidth
    `width - 1` and size `num_contribs * step + width - step`.
    The value of `width` here must be at least `step`.
    """
    assert contribs.shape[1] >= 1 and contribs.shape[1] >= step
    assert contribs.shape[2] == contribs.shape[1]

    cdef unsigned long width = contribs.shape[1]
    cdef unsigned long num_contribs = contribs.shape[0]
    cdef unsigned long depth = contribs.shape[1] - 1
    cdef unsigned long mat_size = num_contribs * step + width - step

    if target_bm is None:
        mat_bm = bm.zeros(depth, depth, mat_size)
    else:
        assert target_bm.l == depth and target_bm.u == depth
        assert target_bm.size == mat_size
        mat_bm = target_bm

    cdef unsigned long transposed = mat_bm.transposed
    cdef cnp.ndarray[cnp.float64_t, ndim=2] mat_data = mat_bm.data

    cdef unsigned long index, frame, k, l

    if transposed:
        for index in range(num_contribs):
            frame = index * step
            for k in range(width):
                for l in range(width):
                    mat_data[depth + l - k, frame + k] += contribs[index, k, l]
    else:
        for index in range(num_contribs):
            frame = index * step
            for k in range(width):
                for l in range(width):
                    mat_data[depth + k - l, frame + l] += contribs[index, k, l]

    if target_bm is None:
        return mat_bm
    else:
        return

@cython.boundscheck(False)
def extract_overlapping_v(cnp.ndarray[cnp.float64_t, ndim=1] vec,
                          unsigned long width,
                          unsigned long step=1,
                          cnp.ndarray[cnp.float64_t, ndim=2] target=None):
    """Extracts overlapping subvectors from a vector.

    The result `subvectors` is a matrix consisting of a sequence of subvectors
    of `vec`.
    Specifically `subvectors[i]` is `vec[(i * step):(i * step + width)]`.

    If `target` is None then a new matrix is returned; otherwise the result is
    written to the matrix `target` (and all elements of `target` are guaranteed
    to be overwritten, so there is no need to zero it ahead of time).
    The length of `vec` should be `num_subs * step + width - step` for some
    `num_subs`, i.e. it should "fit" a whole number of subvectors.
    The returned matrix has shape `(num_subs, width)`.
    The value of `width` here must be at least `step`.
    """
    assert step >= 1
    assert width >= step
    assert vec.shape[0] >= width - step
    assert (vec.shape[0] + step - width) % step == 0

    cdef unsigned long num_subs = (vec.shape[0] + step - width) // step
    cdef cnp.ndarray[cnp.float64_t, ndim=2] subvectors

    if target is None:
        subvectors = np.empty((num_subs, width), dtype=vec.dtype)
    else:
        assert target.shape[0] == num_subs
        assert target.shape[1] == width
        subvectors = target

    cdef unsigned long index, frame, k

    for index in range(num_subs):
        frame = index * step
        for k in range(width):
            subvectors[index, k] = vec[frame + k]

    if target is None:
        return subvectors
    else:
        return

@cython.boundscheck(False)
def extract_overlapping_m(mat_bm,
                          unsigned long step=1,
                          cnp.ndarray[cnp.float64_t, ndim=3] target=None):
    """Extracts overlapping submatrices along the diagonal of a banded matrix.

    The result `submats` is rank-3 tensor consisting of a sequence of
    submatrices from along the diagonal of the matrix represented by `mat_bm`.
    The upper and lower bandwidth of `mat_bm` should be the same.
    Let `width` be `mat_bm.l + 1`.
    If `mat_full` is the matrix represented by `mat_bm` then `submats[i]` is
    `mat_full[(i * step):(i * step + width), (i * step):(i * step + width)]`.

    If `target` is None then a new tensor is returned; otherwise the result is
    written to the tensor `target` (and all elements of `target` are guaranteed
    to be overwritten, so there is no need to zero it ahead of time).
    The size of `mat_bm` should be `num_subs * step + width - step` for some
    `num_subs`, i.e. it should "fit" a whole number of submatrices.
    The returned matrix has shape `(num_subs, width, width)`.
    The value of `width` here must be at least `step`.
    """
    assert mat_bm.l == mat_bm.u

    cdef unsigned long width = mat_bm.l + 1

    assert step >= 1
    assert width >= step
    assert mat_bm.size >= width - step
    assert (mat_bm.size + step - width) % step == 0

    cdef unsigned long num_subs = (mat_bm.size + step - width) // step
    cdef unsigned long depth = mat_bm.l
    cdef unsigned long transposed = mat_bm.transposed
    cdef cnp.ndarray[cnp.float64_t, ndim=2] mat_data = mat_bm.data
    cdef cnp.ndarray[cnp.float64_t, ndim=3] submats

    if target is None:
        submats = np.empty((num_subs, width, width), dtype=mat_data.dtype)
    else:
        assert target.shape[0] == num_subs
        assert target.shape[1] == width
        assert target.shape[2] == width
        submats = target

    cdef unsigned long index, frame, k, l

    if transposed:
        for index in range(num_subs):
            frame = index * step
            for k in range(width):
                for l in range(width):
                    submats[index, k, l] = mat_data[depth + l - k, frame + k]
    else:
        for index in range(num_subs):
            frame = index * step
            for k in range(width):
                for l in range(width):
                    submats[index, k, l] = mat_data[depth + k - l, frame + l]

    if target is None:
        return submats
    else:
        return

def sum_overlapping_v_chunked(contribs_chunks, width, target, step=1):
    """A chunked version of sum_overlapping_v.

    The elements of the iterator `contribs_chunks` should be of the form
    `(start, end, contribs)`.
    For example if `contribs_all` has length 10 then

        sum_overlapping_v_chunked([
            (0, 3, contribs_all[0:3]),
            (3, 10, contribs_all[3:10])
        ])

    produces the same result as `sum_overlapping_v(contribs_all)`.
    This can be used to construct more memory-efficient code in some cases
    (though not in the above example).
    """
    assert step >= 0
    overlap = width - step
    assert overlap >= 0

    for start, end, contribs in contribs_chunks:
        sum_overlapping_v(
            contribs,
            step=step,
            target=target[(start * step):(end * step + overlap)]
        )

def sum_overlapping_m_chunked(contribs_chunks, target_bm, step=1):
    """A chunked version of sum_overlapping_m.

    The elements of the iterator `contribs_chunks` should be of the form
    `(start, end, contribs)`.
    For example if `contribs_all` has length 10 then

        sum_overlapping_m_chunked([
            (0, 3, contribs_all[0:3]),
            (3, 10, contribs_all[3:10])
        ])

    produces the same result as `sum_overlapping_m(contribs_all)`.
    This can be used to construct more memory-efficient code in some cases
    (though not in the above example).
    """
    assert step >= 0
    depth = target_bm.l
    assert target_bm.u == depth
    width = depth + 1
    overlap = width - step
    assert overlap >= 0

    for start, end, contribs in contribs_chunks:
        sum_overlapping_m(
            contribs,
            step=step,
            target_bm=target_bm.sub_matrix_view(
                start * step, end * step + overlap
            )
        )

def extract_overlapping_v_chunked(vec, width, chunk_size, step=1):
    """A chunked version of extract_overlapping_v.

    An iterator over chunks of the output of extract_overlapping_v is returned.
    This can be used to construct more memory-efficient code in some cases.
    """
    assert step >= 1
    overlap = width - step
    assert overlap >= 0
    num_subs = (len(vec) - overlap) // step
    assert num_subs * step + overlap == len(vec)
    assert num_subs >= 0
    assert chunk_size >= 1

    for start in range(0, num_subs, chunk_size):
        end = min(start + chunk_size, num_subs)
        subvectors = extract_overlapping_v(
            vec[(start * step):(end * step + overlap)],
            width,
            step=step
        )
        yield start, end, subvectors

def extract_overlapping_m_chunked(mat_bm, chunk_size, step=1):
    """A chunked version of extract_overlapping_m.

    An iterator over chunks of the output of extract_overlapping_m is returned.
    This can be used to construct more memory-efficient code in some cases.
    """
    assert step >= 1
    depth = mat_bm.l
    assert mat_bm.u == depth
    width = depth + 1
    overlap = width - step
    assert overlap >= 0
    num_subs = (mat_bm.size - overlap) // step
    assert num_subs * step + overlap == mat_bm.size
    assert num_subs >= 0
    assert chunk_size >= 1

    for start in range(0, num_subs, chunk_size):
        end = min(start + chunk_size, num_subs)
        submats = extract_overlapping_m(
            mat_bm.sub_matrix_view(start * step, end * step + overlap),
            step=step
        )
        yield start, end, submats
