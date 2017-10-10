# coding: utf-8
from __future__ import division, print_function, absolute_import

import pkg_resources

from nnmnkwii.datasets import FileDataSource

import numpy as np
from glob import glob
from os.path import join


def example_label_file(phone_level=False):
    """Get path of example HTS-style full-context lable file.

    Corresponding audio file can be accessed by
    :func:`example_audio_file`.

    Args:
        phone_level: If True, returns phone-level aligment, otherwise state-level
          alignment.

    Returns:
        str: Path of the example label file.

    See also:
        :func:`example_audio_file`

    Examples:
        >>> from nnmnkwii.util import example_label_file
        >>> from nnmnkwii.io import hts
        >>> labels = hts.load(example_label_file())
    """
    name = "arctic_a0009"
    label_path = pkg_resources.resource_filename(
        __name__, '_example_data/{}_{}.lab'.format(
            name, "phone" if phone_level else "state"))
    return label_path


def example_audio_file():
    """Get path of audio file.

    Returns:
        str: Path of the example audio file.

    See also:
        :func:`example_label_file`

    Examples:
        >>> from nnmnkwii.util import example_audio_file
        >>> from scipy.io import wavfile
        >>> fs, x = wavfile.read(example_audio_file())
    """
    name = "arctic_a0009"
    wav_path = pkg_resources.resource_filename(
        __name__, '_example_data/{}.wav'.format(name))
    return wav_path


def example_question_file():
    """Get path of example question file.

    The question file was taken from Merlin_.

    .. _Merlin: https://github.com/CSTR-Edinburgh/merlin

    Returns:
        str: Path of the example audio file.

    Examples:
        >>> from nnmnkwii.util import example_question_file
        >>> from nnmnkwii.io import hts
        >>> binary_dict, continuous_dict = hts.load_question_set(example_question_file())
    """
    return pkg_resources.resource_filename(
        __name__, '_example_data/questions-radio_dnn_416.hed')


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
    """Get file data sources for duration model training.

    Returns:
        tuple: Tuple of :obj:`FileDataSource` s for example data.

    Examples:
        >>> from nnmnkwii.util import example_file_data_sources_for_duration_model
        >>> from nnmnkwii.datasets import FileSourceDataset
        >>> X, Y = example_file_data_sources_for_duration_model()
        >>> X, Y = FileSourceDataset(X), FileSourceDataset(Y)
        >>> for x, y in zip(X, Y):
        ...     print(x.shape, y.shape)
        ...
        (35, 416) (35, 5)
        (40, 416) (40, 5)
        (39, 416) (39, 5)
    """
    X = ExampleSLTArcticFileDataSource("X_duration")
    Y = ExampleSLTArcticFileDataSource("Y_duration")

    return X, Y


def example_file_data_sources_for_acoustic_model():
    """Get file data sources for acoustic model training

    Returns:
        tuple: Tuple of :obj:`FileDataSource` s for example data.

    Examples:
        >>> from nnmnkwii.util import example_file_data_sources_for_acoustic_model
        >>> from nnmnkwii.datasets import FileSourceDataset
        >>> X, Y = example_file_data_sources_for_acoustic_model()
        >>> X, Y = FileSourceDataset(X), FileSourceDataset(Y)
        >>> for x, y in zip(X, Y):
        ...     print(x.shape, y.shape)
        ...
        (578, 425) (578, 187)
        (675, 425) (675, 187)
        (606, 425) (606, 187)
    """
    X = ExampleSLTArcticFileDataSource("X_acoustic")
    Y = ExampleSLTArcticFileDataSource("Y_acoustic")

    return X, Y
