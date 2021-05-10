# cython: language_level=3

"""Assorted helpful functions."""

# Copyright 2013, 2014, 2015, 2016, 2017, 2018 Matt Shannon

# This file is part of bandmat.
# See `License` for details of license and warranty.

cimport numpy as cnp
cimport cython

cnp.import_array()
cnp.import_ufunc()

def fancy_plus_equals(cnp.ndarray[cnp.int_t, ndim=1] target_index_seq,
                      cnp.ndarray[cnp.float64_t, ndim=1] source,
                      cnp.ndarray[cnp.float64_t, ndim=1] target):
    """Implements a += method with fancy indexing.

    Does what you might expect
        target[target_index_seq] += source
    to do.
    """
    cdef unsigned long source_size

    source_size = source.shape[0]
    assert target_index_seq.shape[0] == source_size

    cdef unsigned long source_index
    cdef long target_index

    for source_index in range(source_size):
        target_index = target_index_seq[source_index]
        target[target_index] += source[source_index]

    return

def fancy_plus_equals_2d(cnp.ndarray[cnp.int_t, ndim=1] target_index_seq,
                         cnp.ndarray[cnp.float64_t, ndim=2] source,
                         cnp.ndarray[cnp.float64_t, ndim=2] target):
    """Implements a += method with fancy indexing.

    Does what you might expect
        target[target_index_seq] += source
    to do.
    """
    cdef unsigned long source_size
    cdef unsigned long size1

    source_size = source.shape[0]
    assert target_index_seq.shape[0] == source_size
    size1 = source.shape[1]
    assert target.shape[1] == size1

    cdef unsigned long source_index
    cdef long target_index
    cdef unsigned long index1

    for source_index in range(source_size):
        target_index = target_index_seq[source_index]
        for index1 in range(size1):
            target[target_index, index1] += source[source_index, index1]

    return

def fancy_plus_equals_3d(cnp.ndarray[cnp.int_t, ndim=1] target_index_seq,
                         cnp.ndarray[cnp.float64_t, ndim=3] source,
                         cnp.ndarray[cnp.float64_t, ndim=3] target):
    """Implements a += method with fancy indexing.

    Does what you might expect
        target[target_index_seq] += source
    to do.
    """
    cdef unsigned long source_size
    cdef unsigned long size1
    cdef unsigned long size2

    source_size = source.shape[0]
    assert target_index_seq.shape[0] == source_size
    size1 = source.shape[1]
    assert target.shape[1] == size1
    size2 = source.shape[2]
    assert target.shape[2] == size2

    cdef unsigned long source_index
    cdef long target_index
    cdef unsigned long index1
    cdef unsigned long index2

    for source_index in range(source_size):
        target_index = target_index_seq[source_index]
        for index1 in range(size1):
            for index2 in range(size2):
                target[target_index, index1, index2] += (
                    source[source_index, index1, index2]
                )

    return
