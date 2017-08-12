Datasets
========

.. automodule:: nnmnkwii.datasets

This module provides dataset abstraction.
In this library, a dataset represents fixed-sized set of features (e.g., acoustic
features, linguistic features, duration features etc.) composed of multiple
utterances, supporting iteration and indexing.


Interface
----------

To build dataset and to represent variety of features (linguistic, duration,
acoustic, etc) in an unified way, we define couple of interfaces.

1. :obj:`FileDataSource`
2. :obj:`Dataset`

The former is an abstraction of file data sources, where we find the data and
how to process them. Any FileDataSource must implements:

- ``collect_files``: specifies where to find source files (wav, lab, cmp, bin, etc.).
- ``collect_features``: specifies how to collect features (just load from file, or do some feature extraction logic, etc).

The later is an abstraction of dataset. Any dataset must implements
:obj:`Dataset` interface:

- ``__getitem__``: returns features (typically, two dimentional :obj:`numpy.ndarray`)
- ``__len__``: returns the size of dataset (e.g., number of utterances).

One important point is that we use :obj:`numpy.ndarray` to represent features
(there might be exception though). For example,

- F0 trajecoty as ``T x 1`` array, where ``T`` represents number of frames.
- Spectrogram as ``T x D`` array, where ``D`` is number of feature dimention.
- Linguistic features as ``T x D`` array.

.. autoclass:: FileDataSource
    :members:

.. autoclass:: Dataset
    :members:

Implementation
--------------

With combination of :obj:`FileDataSource` and :obj:`Dataset`, we define
some dataset implementation that can be used for typical situations.

.. note::

    Note that we don't provide special iterator implementation (e.g., mini-batch
    iteration, multiprocessing, etc). Users are expected to use dataset with other
    iterator implementation. For PyTorch users, we can use `PyTorch DataLoader`_ for
    mini-batch iteration and multiprocessing. Our dataset interface is `exactly`
    same as PyTorch's one, so we can use PyTorch DataLoader seamlessly.

.. _PyTorch DataLoader: http://pytorch.org/docs/master/data.html?highlight=dataloader#torch.utils.data.DataLoader

Dataset that supports utterance-wise iteration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: FileSourceDataset
    :members:

.. autoclass:: PaddedFileSourceDataset
    :members:

.. autoclass:: MemoryCacheDataset
    :members:


Dataset that supports frame-wise iteration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: MemoryCacheFramewiseDataset
    :members:
