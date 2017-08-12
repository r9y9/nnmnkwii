Autograd
========

.. automodule:: nnmnkwii.autograd

This module provides differenciable functions for PyTorch. This may be extended
to support other autograd frameworks.

Currently all functions doesn't have CUDA implementation, but should be
addressed later.

Functional interface
--------------------

.. autosummary::
    :toctree: generated/

    mlpg
    modspec

Function classes
----------------

.. autoclass:: MLPG
    :members:

.. autoclass:: ModSpec
    :members:
