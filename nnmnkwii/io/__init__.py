"""
IO
==

This modules provides IO operations for some speech-specific file formats.
As of now, it only supports the following files:

- HTS-style question file
- HTS-style full-context label

Note that code for HTS-style formats was initally taken from
merlin/src/label_normalisation.py and refactored to be stateless and
functional.

https://github.com/CSTR-Edinburgh/merlin

HTS IO
------

.. autosummary::
    :toctree: generated/

    load_question_set
    load_label
    extract_durations

"""
from __future__ import division, print_function, absolute_import


from nnmnkwii.io.hts import load_question_set, load_label, extract_durations
