"""
IO
==

This modules provides IO operations for some speech-specific file formats.
As of now, it only supports read operations for:

- HTS-style question file
- HTS-style full-context label file

Note that code for HTS-style formats was initally taken from
merlin/src/label_normalisation.py and refactored to be stateless and
functional.

https://github.com/CSTR-Edinburgh/merlin

.. automodule:: nnmnkwii.io.hts

"""
from __future__ import division, print_function, absolute_import
