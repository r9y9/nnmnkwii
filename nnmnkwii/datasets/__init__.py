from __future__ import with_statement, print_function, absolute_import

import numpy as np

from collections import OrderedDict
from warnings import warn
from tqdm import tqdm


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

        # Multiple files
        if isinstance(collected_files, tuple):
            collected_files = np.asarray(collected_files).T
            lengths = np.array([len(files) for files in collected_files])
            if not (lengths == lengths[0]).all():
                raise RuntimeError(
                    """Mismatch of number of collected files {}.
You must collect same number of files when you collect multiple pair of files.""".format(
                        tuple(lengths)))
        else:
            collected_files = np.atleast_2d(collected_files).T
        if len(collected_files) == 0:
            warn("No files are collected. You might have specified wrong data source.")

        self.collected_files = collected_files

    def __collect_features(self, paths):
        try:
            return self.file_data_source.collect_features(*paths)
        except TypeError as e:
            warn("TypeError while iterating dataset.\n" +
                 "Likely there's mismatch in number of pair of collected files and " +
                 "expected number of arguments of `collect_features`.\n" +
                 "Number of argments: {}\n".format(len(paths)) +
                 "Arguments: {}".format(*paths))
            raise e

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            current, stop, step = idx.indices(len(self))
            return [self[i] for i in range(current, stop, step)]

        paths = self.collected_files[idx]
        return self.__collect_features(paths)

    def __len__(self):
        return len(self.collected_files)

    def asarray(self, padded_length=None, dtype=np.float32,
                padded_length_guess=1000, verbose=0):
        """Convert dataset to numpy array.

        This try to load entire dataset into a single 3d numpy array.

        Args:
            padded_length (int): Number of maximum time frames to be expected.
              If None, it is set to actual maximum time length.
            dtype (numpy.dtype): Numpy dtype.
            padded_length_guess: (int): Initial guess of max time length of
              padded dataset array. Used if ``padded_length`` is None.
        Returns:
            3d-array: Array of shape ``N x T^max x D`` if ``padded_length`` is
            None, otherwise ``N x padded_length x D``.
        """
        collected_files = self.collected_files
        if padded_length is not None:
            T = padded_length
        else:
            T = padded_length_guess  # initial guess

        D = self[0].shape[-1]
        N = len(self)
        X = np.zeros((N, T, D), dtype=dtype)
        lengths = np.zeros(N, dtype=np.int)

        if verbose > 0:
            def custom_range(x):
                return tqdm(range(x))
        else:
            custom_range = range

        for idx in custom_range(len(collected_files)):
            paths = collected_files[idx]
            x = self.__collect_features(paths)
            lengths[idx] = len(x)
            if len(x) > T:
                if padded_length is not None:
                    raise RuntimeError(
                        """Num frames {} exceeded: {}.
Try larger value for padded_length, or set to None""".format(len(x), T))
                warn("Reallocating array because num frames {} exceeded current guess {}.\n".format(
                    len(x), T) +
                    "To avoid memory re-allocations, try large `padded_length_guess` " +
                    "or set `padded_length` explicitly.")
                n = len(x) - T
                # Padd zeros to end of time axis
                X = np.pad(X, [(0, 0), (0, n), (0, 0)],
                           mode="constant", constant_values=0)
                T = X.shape[1]
            X[idx][:len(x), :] = x
            lengths[idx] = len(x)

        if padded_length is None:
            max_len = np.max(lengths)
            X = X[:, :max_len, :]
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

    def asarray(self, dtype=np.float32, verbose=0):
        return super(PaddedFileSourceDataset, self).asarray(
            self.padded_length, dtype=dtype, verbose=verbose)


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
