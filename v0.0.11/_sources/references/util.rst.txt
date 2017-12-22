Utilities
=========

.. automodule:: nnmnkwii.util

Function utilities
------------------

Most of the feature transformation in :obj:`nnmnkwii.preprocessing` module is 2d functions
``f: (T, D) -> (T, D')``. The following utilities can be used for
extending 2d functions to 3d by applying 2d function to each 2d slice.

.. autosummary::
   :toctree: generated/

   apply_each2d_padded
   apply_each2d_trim


Files
-----

Part of files were taken from `CMU ARCTIC dataset`_.
Example quetsion file was taken from Merlin_.

.. _CMU ARCTIC dataset: http://www.festvox.org/cmu_arctic/
.. _Merlin: https://github.com/CSTR-Edinburgh/merlin

.. autosummary::
   :toctree: generated/

   example_label_file
   example_audio_file
   example_question_file
   example_file_data_sources_for_duration_model
   example_file_data_sources_for_acoustic_model


Linear algebra
--------------

.. automodule:: nnmnkwii.util.linalg

.. autosummary::
  :toctree: generated/

  cholesky_inv
  cholesky_inv_banded
