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
        """Collect features given a file path.

        Args:
            args: File path or tuple of file paths

        Returns:
            2darray: ``T`` x ``D`` features represented by 2d array.
        """
        raise NotImplementedError


class DatasetMixIn(object):
    """Dataset represents a fixed-sized set of features.
    """

    def __getitem__(self, idx):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class FileSourceDataset(DatasetMixIn):
    """FileSourceDataset

    Helper to load data from files into array.　This implements __getitem__ and
    __len__ and acts like an array, but it reads data from file on demand.

    TODO:
        Support slice indices

    Attributes:
        file_data_source (FileDataSource): Data source to specify 1) what files
            to be loaded and 2) how to collect features from them.
        collected_files (ndarray): Collected files are stored.

    Args:
        file_data_source (FileDataSource): File data source.
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
        return self.file_data_source.collect_features(*self.collected_files[idx])

    def __len__(self):
        return len(self.collected_files)

    def _get_feature_dim(self, *args, **kwargs):
        x = self.file_data_source.collect_features(*args, **kwargs)
        return x.shape[-1]

    def asarray(self, padded_length, dtype=np.float32):
        """Convert dataset to numpy array.
        This try to load entire dataset into a single 3d numpy array.

        Args:
            max_num_frames (int): Number of maximum time frames to be expected.
        Returns:
            3d-array: ``N`` x ``T`` x ``D`` array
        """
        collected_files = self.collected_files
        T = padded_length

        # Multiple files are collected
        D = self._get_feature_dim(*collected_files[0])
        N = len(self)
        X = np.zeros((N, T, D), dtype=dtype)

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
    def __init__(self, data_source, padded_length):
        super(PaddedFileSourceDataset, self).__init__(data_source)
        self.padded_length = padded_length

    def __getitem__(self, idx):
        x = super(PaddedFileSourceDataset, self).__getitem__(idx)
        if len(x) > self.padded_length:
            raise RuntimeError("""
Num frames {} exceeded: {}. Try larger value for padded_length.""".format(
                len(x), self.padded_length))
        return np.pad(x, [(0, self.padded_length - len(x)), (0, 0)],
                      mode="constant", constant_values=0)

    def asarray(self):
        return super(PaddedFileSourceDataset, self).asarray(self.padded_length)

class MemoryCacheDataset(DatasetMixIn):
    """This is not particulary useful, unless you are indexing with same indices
    multiple times; that's unlikely while we are iterating dataset.
    """
    def __init__(self, dataset, cache_size=100):
        self.dataset = dataset
        self.cached_utterances = OrderedDict()
        self.cache_size = cache_size

    def __getitem__(self, utt_idx):
        if utt_idx not in self.cached_utterances.keys():
            # Load data from file
            self.cached_utterances[utt_idx] = self.dataset[utt_idx]
        if len(self.cached_utterances) > self.cache_size:
            del self.cached_utterances[list(self.cached_utterances.keys())[0]]

        return self.cached_utterances[utt_idx]

    def __len__(self):
        return len(self.dataset)

class MemoryCacheFramewiseDataset(MemoryCacheDataset):
    def __init__(self, dataset, lengths, cache_size=100):
        super(MemoryCacheFramewiseDataset, self).__init__(dataset, cache_size)
        self.lengths = lengths
        self.cumsum_lengths = np.hstack((0, np.cumsum(lengths)))
        self.n_frames = np.sum(lengths)
        assert hasattr(self, "dataset")
        assert hasattr(self, "cached_utterances")
        assert hasattr(self, "cache_size")

    def __getitem__(self, frame_idx):
        # 0-origin
        utt_idx = np.argmax(self.cumsum_lengths > frame_idx) - 1
        frames = super(MemoryCacheFramewiseDataset, self).__getitem__(utt_idx)
        frame_idx_in_focused_utterance = frame_idx - self.cumsum_lengths[utt_idx]
        return frames[frame_idx_in_focused_utterance]

    def __len__(self):
        return self.n_frames
