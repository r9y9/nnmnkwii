from __future__ import with_statement, print_function, absolute_import

import numpy as np

from collections import OrderedDict


class FileDataSource(object):
    """File data source interface.

    Users are expected to implement custum data source for your own data.
    All file data sources must implement this interface.
    """

    def collect_files(self):
        """Collect data source files

        Returns:
            List or tuple of list: List of files, or tuple of list if you need
            multiple files to collect features.
        """
        raise NotImplementedError

    def collect_features(self, *args):
        """Collect features given path(s).

        Args:
            args: File path or tuple of file paths

        Returns:
            2darray: ``T x D`` features represented by 2d array.
        """
        raise NotImplementedError


class Dataset(object):
    """Dataset represents a fixed-sized set of features composed of multiple
    utterances.
    """

    def __getitem__(self, idx):
        """Get access to the dataset.

        Args:
            idx : index

        Returns:
            features
        """
        raise NotImplementedError

    def __len__(self):
        """Length of the dataset

        Returns:
            int: length of dataset. Can be number of utterances or number of
            total frames depends on implementation.
        """
        raise NotImplementedError


class FileSourceDataset(Dataset):
    """FileSourceDataset

    Most basic dataset implementation. It supports utterance-wise iteration and
    has utility (:obj:`asarray` method) to convert dataset to an three
    dimentional :obj:`numpy.ndarray`.

    Speech features have typically different number of time resolusion,
    so we cannot simply represent dataset as an
    array. To address the issue, the dataset class represents set
    of features as ``N x T^max x D`` array by padding zeros where ``N`` is the
    number of utterances, ``T^max`` is maximum number of frame lenghs and ``D``
    is the dimention of features, respectively.

    While this dataset loads features on-demand while indexing, if you are
    dealing with relatively small dataset, it might be useful to convert it to
    an array, and then do whatever with numpy/scipy functionalities.

    Attributes:
        file_data_source (FileDataSource): Data source to specify 1) where to
            find data to be loaded and 2) how to collect features from them.
        collected_files (ndarray): Collected files are stored.

    Args:
        file_data_source (FileDataSource): File data source.

    Examples:
        >>> from nnmnkwii.util import example_file_data_sources_for_acoustic_model
        >>> from nnmnkwii.datasets import FileSourceDataset
        >>> X, Y = example_file_data_sources_for_acoustic_model()
        >>> X, Y = FileSourceDataset(X), FileSourceDataset(Y)
        >>> for (x, y) in zip(X, Y):
        ...     print(x.shape, y.shape)
        ...
        (578, 425) (578, 187)
        (675, 425) (675, 187)
        (606, 425) (606, 187)
        >>> X.asarray(1000).shape
        (3, 1000, 425)
        >>> Y.asarray(1000).shape
        (3, 1000, 187)

    """

    def __init__(self,
                 file_data_source):
        self.file_data_source = file_data_source
        collected_files = self.file_data_source.collect_files()
        if isinstance(collected_files, tuple):
            collected_files = np.asarray(collected_files).T
        else:
            collected_files = np.atleast_2d(collected_files).T
        self.collected_files = collected_files

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            current, stop, step = idx.indices(len(self))
            return [self[i] for i in range(current, stop, step)]
        return self.file_data_source.collect_features(*self.collected_files[idx])

    def __len__(self):
        return len(self.collected_files)

    def asarray(self, padded_length, dtype=np.float32, lengths=None):
        """Convert dataset to numpy array.

        This try to load entire dataset into a single 3d numpy array.

        Args:
            padded_length (int): Number of maximum time frames to be expected.
        Returns:
            3d-array: ``N x T^max x D`` array
        """
        collected_files = self.collected_files
        T = padded_length

        D = self[0].shape[-1]
        N = len(self)
        X = np.zeros((N, T, D), dtype=dtype)

        if lengths is not None:
            assert len(lengths) == N
        else:
            lengths = np.zeros(N, dtype=np.int)
        for idx, paths in enumerate(collected_files):
            x = self.file_data_source.collect_features(*paths)
            if len(x) > T:
                raise RuntimeError("""
Num frames {} exceeded: {}. Try larger value for padded_length.""".format(
                    len(x), T))
                # TODO: segmentation algorithm?
            X[idx][:len(x), :] = x
            lengths[idx] = len(x)
        return X


class PaddedFileSourceDataset(FileSourceDataset):
    """PaddedFileSourceDataset

    Basic dataset with padding. Very similar to :obj:`FileSourceDataset`,
    it supports utterance-wise iteration and has
    utility (:obj:`asarray` method) to convert dataset to an three
    dimentional :obj:`numpy.ndarray`.

    The difference between :obj:`FileSourceDataset` is that this returns
    padded features as ``T^max x D`` array at ``__getitem__``, while
    :obj:`FileSourceDataset` returns not-padded ``T x D`` array.

    Args:
        file_data_source (FileDataSource): File data source.
        padded_length (int): Padded length.

    Attributes:
        file_data_source (FileDataSource)
        padded_length (int)

    Examples:
        >>> from nnmnkwii.util import example_file_data_sources_for_acoustic_model
        >>> from nnmnkwii.datasets import PaddedFileSourceDataset
        >>> X.asarray(1000).shape
        (3, 1000, 425)
        >>> X, Y = example_file_data_sources_for_acoustic_model()
        >>> X, Y = PaddedFileSourceDataset(X, 1000), PaddedFileSourceDataset(Y, 1000)
        >>> for (x, y) in zip(X, Y):
        ...     print(x.shape, y.shape)
        ...
        (1000, 425) (1000, 187)
        (1000, 425) (1000, 187)
        (1000, 425) (1000, 187)
        >>> X.asarray().shape
        (3, 1000, 425)
        >>> Y.asarray().shape
        (3, 1000, 187)
    """

    def __init__(self, file_data_source, padded_length):
        super(PaddedFileSourceDataset, self).__init__(file_data_source)
        self.padded_length = padded_length

    def _getitem_one_sample(self, idx):
        x = super(PaddedFileSourceDataset, self).__getitem__(idx)
        if len(x) > self.padded_length:
            raise RuntimeError("""
Num frames {} exceeded: {}. Try larger value for padded_length.""".format(
                len(x), self.padded_length))
        return np.pad(x, [(0, self.padded_length - len(x)), (0, 0)],
                      mode="constant", constant_values=0)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            current, stop, step = idx.indices(len(self))
            xs = [self._getitem_one_sample(i)
                  for i in range(current, stop, step)]
            return np.array(xs)
        else:
            return self._getitem_one_sample(idx)

    def asarray(self, dtype=np.float32, lengths=None):
        return super(PaddedFileSourceDataset, self).asarray(
            self.padded_length, dtype=dtype, lengths=lengths)


class MemoryCacheDataset(Dataset):
    """MemoryCacheDataset

    A thin dataset wrapper class that has simple cache functionality. It supports
    utterance-wise iteration.

    Args:
        dataset (Dataset): Dataset implementation to wrap.
        cache_size (int): Cache size (utterance unit).

    Attributes:
        dataset (Dataset): Dataset
        cached_utterances (OrderedDict): Loaded utterances. Keys are utterance
          indices and values are numpy arrays.
        cache_size (int): Cache size.

    Examples:
        >>> from nnmnkwii.util import example_file_data_sources_for_acoustic_model
        >>> from nnmnkwii.datasets import FileSourceDataset
        >>> X, Y = example_file_data_sources_for_acoustic_model()
        >>> X, Y = FileSourceDataset(X), FileSourceDataset(Y)
        >>> from nnmnkwii.datasets import MemoryCacheDataset
        >>> X, Y = MemoryCacheDataset(X), MemoryCacheDataset(Y)
        >>> X.cached_utterances
        OrderedDict()
        >>> for (x, y) in zip(X, Y):
        ...     print(x.shape, y.shape)
        ...
        (578, 425) (578, 187)
        (675, 425) (675, 187)
        (606, 425) (606, 187)
        >>> len(X.cached_utterances)
        3
    """

    def __init__(self, dataset, cache_size=777):
        self.dataset = dataset
        self.cached_utterances = OrderedDict()
        self.cache_size = cache_size

    def __getitem__(self, utt_idx):
        if utt_idx not in self.cached_utterances.keys():
            # Load data from file
            self.cached_utterances[utt_idx] = self.dataset[utt_idx]
        if len(self.cached_utterances) > self.cache_size:
            self.cached_utterances.popitem(last=False)

        return self.cached_utterances[utt_idx]

    def __len__(self):
        return len(self.dataset)


class MemoryCacheFramewiseDataset(MemoryCacheDataset):
    """MemoryCacheFramewiseDataset

    A thin dataset wrapper class that has simple cache functionality. It supports
    frame-wise iteration. Different from other utterance-wise datasets, you will
    need to explicitly give number of time frames for each utterance at
    construction, since the class has to know the size of dataset to implement
    ``__len__``.

    Note:
        If you are doing random access to the dataset, please be careful that you
        give sufficient large number of cache size, to avoid many file re-loading.

    Args:
        dataset (Dataset): Dataset implementation to wrap.
        lengths (list): Frame lengths for each utterance.
        cache_size (int): Cache size (utterance unit).

    Attributes:
        dataset (Dataset): Dataset
        cached_utterances (OrderedDict): Loaded utterances.
        cache_size (int): Cache size.

    Examples
        >>> from nnmnkwii.util import example_file_data_sources_for_acoustic_model
        >>> from nnmnkwii.datasets import FileSourceDataset
        >>> from nnmnkwii.datasets import MemoryCacheFramewiseDataset
        >>> X, Y = example_file_data_sources_for_acoustic_model()
        >>> X, Y = FileSourceDataset(X), FileSourceDataset(Y)
        >>> len(X)
        3
        >>> lengths = [len(x) for x in X] # collect frame lengths
        >>> X = MemoryCacheFramewiseDataset(X, lengths)
        >>> Y = MemoryCacheFramewiseDataset(Y, lengths)
        >>> len(X)
        1859
        >>> x[0].shape
        (425,)
        >>> y[0].shape
        (187,)
    """

    def __init__(self, dataset, lengths, cache_size=777):
        super(MemoryCacheFramewiseDataset, self).__init__(dataset, cache_size)
        self.lengths = lengths
        self.cumsum_lengths = np.hstack((0, np.cumsum(lengths)))
        self.n_frames = np.sum(lengths)
        assert hasattr(self, "dataset")
        assert hasattr(self, "cached_utterances")
        assert hasattr(self, "cache_size")

    def _getitem_one_sample(self, frame_idx):
        # 0-origin
        utt_idx = np.argmax(self.cumsum_lengths > frame_idx) - 1
        frames = super(MemoryCacheFramewiseDataset, self).__getitem__(utt_idx)
        frame_idx_in_focused_utterance = frame_idx - \
            self.cumsum_lengths[utt_idx]
        return frames[frame_idx_in_focused_utterance]

    def __getitem__(self, frame_idx):
        if isinstance(frame_idx, slice):
            current, stop, step = frame_idx.indices(len(self))
            xs = [self._getitem_one_sample(i)
                  for i in range(current, stop, step)]
            return np.array(xs)
        else:
            return self._getitem_one_sample(frame_idx)

    def __len__(self):
        return self.n_frames
