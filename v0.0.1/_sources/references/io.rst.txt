IO
==

IO operations for some speech-specific file formats.
As of now, it only supports read operations for:

- HTS-style question file
- HTS-style full-context label file

HTS IO
------

.. automodule:: nnmnkwii.io.hts

.. autosummary::
    :toctree: generated/

    load
    load_question_set

.. autoclass:: HTSLabelFile
    :members:
