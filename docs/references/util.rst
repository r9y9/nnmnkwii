Utilities
=========

.. automodule:: nnmnkwii.util

Utterance-wise operations
-------------------------

.. autosummary::
   :toctree: generated/

   delta
   apply_delta_windows
   trim_zeros_frames
   remove_zeros_frames
   adjast_frame_length
   scale
   minmax_scale

Dataset-wise operations
-----------------------

.. autosummary::
   :toctree: generated/

   meanvar
   meanstd
   minmax

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
