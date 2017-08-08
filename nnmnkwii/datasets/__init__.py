from __future__ import with_statement, print_function, absolute_import

import numpy as np


class DataSource(object):
    """Data source interface.

    Users are expected to implement custum data source for your own data.
    All data sources must implement this interface.
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
    def __init__(self, data_source):
        self.data_source = data_source

    def __getitem__(self, idx):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class Dataset(DatasetMixIn):
    """Dataset

    Helper to load data into array.

    Attributes:
        data_source (DataSource): Data source to specify 1) what files to be
            loaded and 2) how to collect features from them.
        collected_files (ndarray): Collected files are stored.
    """

    def __init__(self,
                 data_source):
        super(Dataset, self).__init__(data_source)
        collected_files = self.data_source.collect_files()
        if isinstance(collected_files, tuple):
            collected_files = np.asarray(collected_files).T
        else:
            collected_files = np.atleast_2d(collected_files).T
        self.collected_files = collected_files

    def __getitem__(self, idx):
        x = self.data_source.collect_features(*self.collected_files[idx])
        return x

    def __len__(self):
        return len(self.collected_files)

    def _get_feature_dim(self, *args, **kwargs):
        x = self.data_source.collect_features(*args, **kwargs)
        return x.shape[-1]

    def asarray(self, max_num_frames=1000):
        """Convert dataset to numpy array.

        This try to load entire dataset into a single 3d numpy array.

        Args:
            max_num_frames (int): Number of maximum time frames to be expected.

        Returns:
            3d-array: ``N`` x ``T`` x ``D`` array
        """
        collected_files = self.collected_files
        T = max_num_frames

        # Multiple files are collected
        D = self._get_feature_dim(*collected_files[0])
        N = len(self)
        X = np.zeros((N, T, D), dtype=np.float32)

        lengths = np.zeros(N, dtype=np.int)
        for idx, paths in enumerate(collected_files):
            x = self.data_source.collect_features(*paths)
            if len(x) > max_num_frames:
                raise RuntimeError("""
Num frames {} exceeded: {}. Try larger value for max_num_frames.""".format(
                    len(x), max_num_frames))
                # TODO: segmentation algorithm?
            X[idx][:len(x), :] = x
            lengths[idx] = len(x)
        return X


# TODO: Generic Dataloader
# shuffle, multiprocessing, like pytorch?

class LabeledDataset(Dataset):
    def __init__(self,
                 data_source):
        super(LabeledDataset, self).__init__(data_source)
        if self.labels is None:
            raise RuntimeError("You need to collect labels in your DataSource.")

    def __getitem__(self, idx):
        features = self.data_source.collect_features(self.paths[idx])
        return features, self.labels[idx]

    def asarray(self, max_num_frames=1000):
        X = super(LabeledDataset, self).asarray(max_num_frames)
        Y = np.asarray(self.labels).astype(np.int)
        return X, Y
