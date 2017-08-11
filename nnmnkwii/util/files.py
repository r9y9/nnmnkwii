"""
Files
=====

"""
from __future__ import division, print_function, absolute_import

from nnmnkwii.datasets import FileDataSource

from sklearn.model_selection import train_test_split
from os.path import join
from glob import glob
import numpy as np

import pkg_resources

__all__ = [
    "example_label_file",
    "example_audio_file",
    "example_linguistic_acoustic_pairs_file",
    "ExampleLinguisticFileDataSource",
    "ExampleAcousticFileDataSource",
]


def example_label_file():
    name = "arctic_a0009"
    label_path = pkg_resources.resource_filename(
        __name__, '_example_data/{}_state.lab'.format(name))
    return label_path


def example_audio_file():
    name = "arctic_a0009"
    wav_path = pkg_resources.resource_filename(
        __name__, '_example_data/{}.wav'.format(name))
    return wav_path


def example_linguistic_acoustic_pairs_file():
    """Get file path of example linguistic/acoustic features path.

    Returns:
        str: File path of example linguistic/acoustic features.

    """
    return pkg_resources.resource_filename(__name__, '_example_data/foobar.npz')


class ExampleLinguisticFileDataSource(FileDataSource):
    DATA_ROOT = pkg_resources.resource_filename(
        __name__, '_example_data/slt_arctic_demo_data/X_linguistic')

    def __init__(self, train=True, test_size=0.1):
        self.train = train
        self.test_size = test_size

    def collect_files(self):
        files = sorted(glob(join(self.DATA_ROOT, "*.npz")))
        train_files, test_files = train_test_split(
            files, test_size=self.test_size, random_state=1234)
        if self.train:
            return train_files
        else:
            return test_files

    def collect_features(self, path):
        return np.load(path)["x"].reshape(-1, 425)


class ExampleAcousticFileDataSource(FileDataSource):
    DATA_ROOT = pkg_resources.resource_filename(
        __name__, '_example_data/slt_arctic_demo_data/Y_acoustic')

    mgc_dim = 75
    lf0_dim = 3
    vuv_dim = 1
    bap_dim = 3

    fs = 16000
    frame_period = 5
    hop_length = 80
    fftlen = 1024
    alpha = 0.41

    mgc_start_idx = 0
    lf0_start_idx = 75
    vuv_start_idx = 78
    bap_start_idx = 79

    windows = [
        (0, 0, np.array([1.0])),
        (1, 1, np.array([-0.5, 0.0, 0.5])),
        (1, 1, np.array([1.0, -2.0, 1.0])),
    ]

    def __init__(self, train=True, test_size=0.1):
        self.train = train
        self.test_size = test_size

    def collect_files(self):
        files = sorted(glob(join(self.DATA_ROOT, "*.npz")))
        train_files, test_files = train_test_split(
            files, test_size=self.test_size, random_state=1234)
        if self.train:
            return train_files
        else:
            return test_files

    def collect_features(self, path):
        return np.load(path)["y"].reshape(
            -1, self.mgc_dim + self.lf0_dim + self.vuv_dim + self.bap_dim)
