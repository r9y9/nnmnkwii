from __future__ import with_statement, print_function, absolute_import

from nnmnkwii.datasets import FileDataSource

import numpy as np
from os.path import join, splitext, isdir
from os import listdir

# List of available speakers.
_speakers = ["awb", "bdl", "clb", "jmk", "ksp", "rms", "slt"]


def _name_to_dirname(name):
    assert len(name) == 3
    return join("cmu_us_{}_arctic".format(name), "wav")


class CMUArcticWavFileDataSource(FileDataSource):
    """File data source for CMU Arctic dataset.

    A builtin implementation of ``FileDataSource`` for CMU Arctic dataset.
    Users are expected to inherit the class and implement ``collect_features``
    method, which defines how features are computed given a wav path.

    Args:
        data_root (str): Data root.
        speakers (list): List of speakers to find. Supported names of speaker
         are "awb", "bdl", "clb", "jmk", "ksp", "rms" and "slt".
        labelmap (dict[optional]): Dict of speaker labels. If None,
          it's assigned as incrementally (i.e., 0, 1, 2) for specified
          speakers.
        max_files (int): Maximum files per to load for each speaker.

    Attributes:
        labels (numpy.ndarray): Speaker labels paired with collected files.
          Stored in ``collect_files``. This is useful to build multi-speaker
          models.
    """

    def __init__(self, data_root, speakers, labelmap=None, max_files=2):
        for speaker in speakers:
            if speaker not in _speakers:
                raise ValueError(
                    "Unknown speaker '{}'. It should be one of {}".format(
                        speaker, _speakers))

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
        speaker_dirs = list(
            map(lambda x: join(self.data_root, _name_to_dirname(x)),
                self.speakers))
        paths = []
        labels = []
        for (i, d) in enumerate(speaker_dirs):
            if not isdir(d):
                raise RuntimeError("{} doesn't exist.".format(d))
            files = [join(speaker_dirs[i], f) for f in listdir(d)]
            files = list(filter(lambda x: splitext(x)[1] == ".wav", files))
            files = sorted(files)
            files = files[:self.max_files]
            for f in files[:self.max_files]:
                paths.append(f)
                labels.append(self.labelmap[self.speakers[i]])

        self.labels = np.array(labels, dtype=np.int32)
        return paths
