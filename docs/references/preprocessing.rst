Pre-processing
==============

Pre-processing algorithms. Mostly, it consists of feature transformation,
feature alignment and feature normalization.

.. automodule:: nnmnkwii.preprocessing

Generic
-------

Utterance-wise operations
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/

   delta_features
   trim_zeros_frames
   remove_zeros_frames
   adjast_frame_length
   scale
   minmax_scale


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
