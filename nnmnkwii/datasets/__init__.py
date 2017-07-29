"""
Dataset abstractions
====================

.. autoclass:: DataSource
    :members:

.. autoclass:: Dataset
    :members:

.. autoclass:: BatchDataset
    :members:

.. autoclass:: IncrementalDataset
    :members:

"""

from __future__ import with_statement, print_function, absolute_import

import numpy as np


class DataSource(object):
    def collect_files(self):
        """Collect data source files
        """
        raise NotImplementedError

    def process_file(self, path):
        """Process file

        This should be called while processing collected data.
        """
        raise NotImplementedError


class Dataset(object):
    def __init__(self, data_source):
        self.data_source = data_source

    def __getitem__(self, idx):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class BatchDataset(Dataset):
    """Dataset that loads entire data into memory
    """
    def __init__(self,
                 data_source,
                 framewise=True,
                 max_num_frames=1000):
        super(BatchDataset, self).__init__(data_source)
        self.framewise = framewise
        self.num_frames = 0
        self.max_num_frames = max_num_frames
        self.X = None
        self.Y = None
        self.D = None

    def __getitem__(self, idx):
        if self.framewise:
            n, frame_idx = self._utterance_wise_idx_to_frame_wise_idx(idx)
            return self.X[n][frame_idx], self.Y[idx]
        else:
            return self.X[idx], self.Y[idx]

    def __len__(self):
        if self.framewise:
            return self.num_frames
        else:
            if self.X is None:
                return 0
            else:
                return self.X.shape[0]

    def _get_feature_dim(self, path):
        x = self.data_source.process_file(path)
        return x.shape[-1]

    def _utterance_wise_idx_to_frame_wise_idx(self, idx):
        T = self.X.shape[1]
        return divmod(idx, T)

    def load(self):
        paths, labels = self.data_source.collect_files()
        labels = np.asarray(labels)
        assert len(paths) > 0
        T = self.max_num_frames
        self.D = self._get_feature_dim(paths[0])
        N = len(paths)
        self.num_frames = N * T

        X = np.zeros((N, T, self.D), dtype=np.float32)
        for idx, path in enumerate(paths):
            x = self.data_source.process_file(path)
            # TODO: segmentation algorithm
            if len(x) > self.max_num_frames:
                raise RuntimeError("""
Num frames {} exceeded: {}. Try larger value for max_num_frames.""".format(
                    len(x), self.max_num_frames))
            X[idx][:len(x), :] = x

        self.X = X
        self.Y = labels.astype(np.int)

        return self.X, self.Y


class IncrementalDataset(Dataset):
    def __init__(self):
        raise NotImplementedError
