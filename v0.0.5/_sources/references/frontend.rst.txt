Frontend
========

Utilities to convert structural representation that has rich
linguistic contexts (e.g, HTS-style full-context label) to its numerical
form. Note that the module doesn't mean to cover language-dependent text
processing frontend.

Merlin frontend
---------------

The code here was initally taken from `Merlin`_'s `label_normalisation.py`_ and
refactored to be stateless and functional APIs.

.. _label_normalisation.py: https://goo.gl/AJaxCa
.. _Merlin: https://github.com/CSTR-Edinburgh/merlin

.. automodule:: nnmnkwii.frontend.merlin

.. autosummary::
    :toctree: generated/

    linguistic_features
    duration_features
