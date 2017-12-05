from __future__ import with_statement, print_function, absolute_import

from nnmnkwii.datasets import FileDataSource

import numpy as np
from os.path import join, splitext, isdir
from os import listdir

# List of available speakers.
available_speakers = ["awb", "bdl", "clb", "jmk", "ksp", "rms", "slt"]


def _name_to_dirname(name):
    assert len(name) == 3
    return join("cmu_us_{}_arctic".format(name), "wav")


class WavFileDataSource(FileDataSource):
    """Wav file data source for CMU Arctic dataset.

    The data source collects wav files from CMU Arctic.
    Users are expected to inherit the class and implement ``collect_features``
    method, which defines how features are computed given a wav file path.

    Args:
        data_root (str): Data root.
        speakers (list): List of speakers to find. Supported names of speaker
         are ``awb``, ``bdl``, ``clb``, ``jmk``, ``ksp``, ``rms`` and ``slt``.
        labelmap (dict[optional]): Dict of speaker labels. If None,
          it's assigned as incrementally (i.e., 0, 1, 2) for specified
          speakers.
        max_files (int): Total number of files to be collected.

    Attributes:
        labels (numpy.ndarray): Speaker labels paired with collected files.
          Stored in ``collect_files``. This is useful to build multi-speaker
          models.
    """

    def __init__(self, data_root, speakers, labelmap=None, max_files=None):
        for speaker in speakers:
            if speaker not in available_speakers:
                raise ValueError(
                    "Unknown speaker '{}'. It should be one of {}".format(
                        speaker, available_speakers))

        self.data_root = data_root
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
        speaker_dirs = list(
            map(lambda x: join(self.data_root, _name_to_dirname(x)),
                self.speakers))
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
            for f in files:
                paths.append(f)
                labels.append(self.labelmap[self.speakers[i]])

        self.labels = np.array(labels, dtype=np.int32)
        return paths


# For compat, remove this after v0.1.0
CMUArcticWavFileDataSource = WavFileDataSource
