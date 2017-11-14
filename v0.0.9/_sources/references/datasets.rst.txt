Datasets
========

.. automodule:: nnmnkwii.datasets

This module provides dataset abstraction.
In this library, a dataset represents fixed-sized set of features (e.g., acoustic
features, linguistic features, duration features etc.) composed of multiple
utterances, supporting iteration and indexing.


Interface
----------

To build dataset and represent variety of features (linguistic, duration,
acoustic, etc) in an unified way, we define couple of interfaces.

1. :obj:`FileDataSource`
2. :obj:`Dataset`

The former is an abstraction of file data sources, where we find the data and
how to process them. Any FileDataSource must implement:

- ``collect_files``: specifies where to find source files (wav, lab, cmp, bin, etc.).
- ``collect_features``: specifies how to collect features (just load from file, or do some feature extraction logic, etc).

The later is an abstraction of dataset. Any dataset must implement
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
    same as PyTorch's one, so we can use PyTorch DataLoader seamlessly. See
    tutorials how we can use it practically.

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


Builtin data sources
--------------------

.. warning::
    builtin data sources are experimental features. May change in future.

There are a couple of builtin file data sources for typical datasets to make it
easy to work on those. With the following data source implementation,
you only need to implement ``collect_features``, which
defines what features you want from wav file or text (depends on data source).
If you want maximum flexibility to access dataset, you may want to implement your
own data source, instead of using bulitin ones.

Suppose we are trying to extract acoustic features from wav files from
CMU Arctic, then you can write for example:

.. code-block:: python

    from nnmnkwii.preprocessing import trim_zeros_frames
    from nnmnkwii.datasets import FileSourceDataset
    from nnmnkwii.datasets import cmu_arctic
    import pysptk
    import pyworld

    class MyFileDataSource(cmu_arctic.WavFileDataSource):
        def __init__(self, data_root, speakers, max_files=100):
            super(MyFileDataSource, self).__init__(
                data_root, speakers, max_files=100)

        def collect_features(self, path):
            """Compute mel-cepstrum given a wav file."""
            fs, x = wavfile.read(path)
            x = x.astype(np.float64)
            f0, timeaxis = pyworld.dio(x, fs, frame_period=5)
            f0 = pyworld.stonemask(x, f0, timeaxis, fs)
            spectrogram = pyworld.cheaptrick(x, f0, timeaxis, fs)
            spectrogram = trim_zeros_frames(spectrogram)
            mc = pysptk.sp2mc(spectrogram, order=24, alpha=0.41)
            return mc.astype(np.float32)

    DATA_ROOT = "/home/ryuichi/data/cmu_arctic/" # your data path
    data_source = MyFileDataSource(DATA_DIR, speakers=["clb"], max_files=100)

    # 100 wav files of `clb` speaker will be collected
    X = FileSourceDataset(data_source)
    assert len(X) == 100

    for x in X:
        # do anything on acoustic features (e.g., save to disk)
        pass

More real examples can be found in `tests directory`_ in nnmnkwii and
tutorial notebooks in `nnmnkwii_gallery`_.

.. _`tests directory`: https://github.com/r9y9/nnmnkwii/tree/master/tests
.. _`nnmnkwii_gallery`: https://github.com/r9y9/nnmnkwii_gallery

CMU Arctic (en)
^^^^^^^^^^^^^^^

You can download data from http://festvox.org/cmu_arctic/.

.. autoclass:: nnmnkwii.datasets.cmu_arctic.WavFileDataSource
    :members:

VCTK (en)
^^^^^^^^^

You can download data (15GB) from http://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html.

.. note::

    Note that VCTK data sources don't collect files for speaker ``315``, since there
    are no transcriptions available for ``315`` entries,

.. autoclass:: nnmnkwii.datasets.vctk.TranscriptionDataSource
    :members:

.. autoclass:: nnmnkwii.datasets.vctk.WavFileDataSource
    :members:


LJ-Speech (en)
^^^^^^^^^^^^^^

You can download data (2.6GB) from https://keithito.com/LJ-Speech-Dataset/.

.. autoclass:: nnmnkwii.datasets.ljspeech.TranscriptionDataSource
    :members:

.. autoclass:: nnmnkwii.datasets.ljspeech.NormalizedTranscriptionDataSource
    :members:

.. autoclass:: nnmnkwii.datasets.ljspeech.WavFileDataSource
    :members:

Voice Conversion Challenge (VCC) 2016 (en)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can download training data (181MB) and evaluation data (~56 MB) from http://datashare.is.ed.ac.uk/handle/10283/2211.

.. autoclass:: nnmnkwii.datasets.vcc2016.WavFileDataSource
    :members:

Voice statistics (jp)
^^^^^^^^^^^^^^^^^^^^^

You can download data (~720MB) from https://voice-statistics.github.io/.

.. autoclass:: nnmnkwii.datasets.voice_statistics.WavFileDataSource
    :members:


JSUT (jp)
^^^^^^^^^

JSUT (Japanese speech corpus of Saruwatari Lab, University of Tokyo).

You can download data (2.7GB) from https://sites.google.com/site/shinnosuketakamichi/publication/jsut.

.. warning::
    As of Nov. 4, 2017, 30 wav files are missing in just_ver1, while transcriptions exist.
    Note that current data source implementations do ignore these missing files.

.. autoclass:: nnmnkwii.datasets.jsut.TranscriptionDataSource
    :members:

.. autoclass:: nnmnkwii.datasets.jsut.WavFileDataSource
    :members:
