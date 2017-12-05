# coding: utf-8
from __future__ import with_statement, print_function, absolute_import

from nnmnkwii.datasets import FileDataSource

import numpy as np
from os.path import join, splitext, isdir
from os import listdir

# List of available speakers.
available_speakers = [
    "SF1", "SF2", "SF3", "SM1", "SM2", "TF1", "TF2", "TM1", "TM2", "TM3"]


class WavFileDataSource(FileDataSource):
    """Wav file data source for Voice Conversion Challenge (VCC) 2016 dataset.

    The data source collects wav files from VCC2016 dataset.
    Users are expected to inherit the class and implement ``collect_features``
    method, which defines how features are computed given a wav file path.

    .. note::
        VCC2016 datasets are composed of training data and evaluation data,
        which can be downloaded separately. ``data_root`` should point to the
        directory that contains both the training and evaluation data.


    Directory structure should look like for example:

    .. code-block:: shell

        > tree -d ~/data/vcc2016/
        /home/ryuichi/data/vcc2016/
        ├── evaluation_all
        │   ├── SF1
        │   ├── SF2
        │   ├── SF3
        │   ├── SM1
        │   ├── SM2
        │   ├── TF1
        │   ├── TF2
        │   ├── TM1
        │   ├── TM2
        │   └── TM3
        └── vcc2016_training
            ├── SF1
            ├── SF2
            ├── SF3
            ├── SM1
            ├── SM2
            ├── TF1
            ├── TF2
            ├── TM1
            ├── TM2
            └── TM3

    Args:
        data_root (str): Data root. It's assumed that training and evaluation
          data are placed at ``${data_root}/vcc2016_training`` and
        ``${data_root}/evaluation_all``, respectively, by default.
        speakers (list): List of speakers to find. Supported names of speaker
         are ``SF1``, ``SF2``, ``SF3``, ``SM1``, ``SM2``, ``TF1``, ``TF2``,
         ``TM1``, ``TM2`` and ``TM3``.
        labelmap (dict[optional]): Dict of speaker labels. If None,
          it's assigned as incrementally (i.e., 0, 1, 2) for specified
          speakers.
        max_files (int): Total number of files to be collected.
        training_data_root: If specified, try to search training data to the
          directory. If None, set to ``${data_root}/vcc2016_training``.
        evaluation_data_root: If specified, try to search evaluation data to the
          directory. If None, set to ``${data_root}/evaluation_all``.
        training (bool): Whether it collects training data or not. If False,
          it collects evaluation data.

    Attributes:
        labels (numpy.ndarray): Speaker labels paired with collected files.
          Stored in ``collect_files``. This is useful to build multi-speaker
          models.
    """

    def __init__(self, data_root, speakers, labelmap=None, max_files=None,
                 training_data_root=None,
                 evaluation_data_root=None,
                 training=True):
        if training_data_root is None:
            training_data_root = join(data_root, "vcc2016_training")
        if evaluation_data_root is None:
            evaluation_data_root = join(data_root, "evaluation_all")

        for speaker in speakers:
            if speaker not in available_speakers:
                raise ValueError(
                    "Unknown speaker '{}'. It should be one of {}".format(
                        speaker, available_speakers))

        self.data_root = data_root
        self.training_data_root = training_data_root
        self.evaluation_data_root = evaluation_data_root
        self.training = training
        self.speakers = speakers
        if labelmap is None:
            labelmap = {}
            for idx, speaker in enumerate(speakers):
                labelmap[speaker] = idx
        self.labelmap = labelmap
        self.max_files = max_files
        self.labels = None

    def collect_files(self):
        """Collect wav files for specific speakers.

        Returns:
            list: List of collected wav files.
        """
        data_root = self.training_data_root if self.training else \
            self.evaluation_data_root
        speaker_dirs = list(map(lambda x: join(data_root, x), self.speakers))
        paths = []
        labels = []

        if self.max_files is None:
            max_files_per_speaker = None
        else:
            max_files_per_speaker = self.max_files // len(self.speakers)
        for (i, d) in enumerate(speaker_dirs):
            if not isdir(d):
                raise RuntimeError("{} doesn't exist.".format(d))
            files = [join(speaker_dirs[i], f) for f in listdir(d)]
            files = list(filter(lambda x: splitext(x)[1] == ".wav", files))
            files = sorted(files)
            files = files[:max_files_per_speaker]
            for f in files[:max_files_per_speaker]:
                paths.append(f)
                labels.append(self.labelmap[self.speakers[i]])

        self.labels = np.array(labels, dtype=np.int32)
        return paths
