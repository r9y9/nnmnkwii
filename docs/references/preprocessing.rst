Pre-processing
==============

Feature transformation, feature alignment and feature normalization.

.. automodule:: nnmnkwii.preprocessing

Generic
-------

Utterance-wise operations
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/

   preemphasis
   inv_preemphasis
   delta_features
   trim_zeros_frames
   remove_zeros_frames
   adjast_frame_length
   adjast_frame_lengths
   scale
   inv_scale
   minmax_scale_params
   minmax_scale
   inv_minmax_scale


Dataset-wise operations
^^^^^^^^^^^^^^^^^^^^^^^
.. autosummary::
   :toctree: generated/

   meanvar
   meanstd
   minmax


F0
--

F0-specific pre-processsing algorithms.

.. automodule:: nnmnkwii.preprocessing.f0

.. autosummary::
   :toctree: generated/

   interp1d


Alignment
---------

Alignment algorithms. This is typically useful for creating parallel data in
statistical voice conversion.

Currently, there are only high-level APIs that takes input as tuple of
unnormalized padded data arrays ``(N x T x D)``
and returns padded aligned arrays with the same shape. If you are interested
in aligning *single* pair of feature matrix (not dataset), then use fastdtw_
directly instead.

.. _fastdtw: https://github.com/slaypni/fastdtw

.. automodule:: nnmnkwii.preprocessing.alignment

.. autoclass:: DTWAligner
 :members:

.. autoclass:: IterativeDTWAligner
 :members:
