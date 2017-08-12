from __future__ import division, print_function, absolute_import

import pkg_resources

from nnmnkwii.datasets import FileDataSource

import numpy as np
from glob import glob
from os.path import join


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


class BinaryFileDataSource(FileDataSource):
    def __init__(self, data_root):
        self.data_root = data_root

    def collect_files(self):
        return sorted(glob(join(self.data_root, "*.npz")))

    def collect_features(self, path):
        return np.load(path)["data"]


class ExampleSLTArcticFileDataSource(BinaryFileDataSource):
    SLT_DEMO_DATA_ROOT = pkg_resources.resource_filename(
        __name__, '_example_data/slt_arctic_demo_data')

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

    def __init__(self, directory):
        super(ExampleSLTArcticFileDataSource, self).__init__(
            join(self.SLT_DEMO_DATA_ROOT, directory))


def example_file_data_sources_for_duration_model():
    X = ExampleSLTArcticFileDataSource("X_duration")
    Y = ExampleSLTArcticFileDataSource("Y_duration")

    return X, Y


def example_file_data_sources_for_acoustic_model():
    X = ExampleSLTArcticFileDataSource("X_acoustic")
    Y = ExampleSLTArcticFileDataSource("Y_acoustic")

    return X, Y
