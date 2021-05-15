IO
==

IO operations for some speech-specific file formats.

- HTS-style full-context label file (a.k.a. HTK alignment)
- HTS-style question file

HTS IO
------

.. automodule:: nnmnkwii.io.hts

.. autosummary::
    :toctree: generated/

    load
    load_question_set
    write_audacity_labels
    write_textgrid

.. autoclass:: HTSLabelFile
    :members:
