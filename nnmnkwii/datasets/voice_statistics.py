from __future__ import with_statement, print_function, absolute_import

from nnmnkwii.datasets import FileDataSource

import numpy as np
from os.path import join, splitext, isdir
from os import listdir

_speakers = ["fujitou", "tsuchiya", "uemura"]
_emotions = ["angry", "happy", "normal"]


def _get_dir(speaker, emotion):
    return "{}_{}".format(speaker, emotion)


class VoiceStatisticsWavFileDataSource(FileDataSource):
    """File data source for Voice-statistics dataset.

    A builtin implementation of ``FileDataSource`` for voice-statistics
    dataset. Users are expected to inherit the class and implement
    ``collect_features`` method, which defines how features are computed
    given a wav path.

    You can download data (~720MB) from `voice-statistics`_.

    _voice-statistics: http://voice-statistics.github.io/

    Args:
        data_root (str): Data root
        speakers (list): List of speakers to load. Supported names of speaker
          are "fujitou", "tsuchiya" and "uemura".
        labelmap (dict[optional]): Dict of speaker labels. If None,
          it's assigned as incrementally (i.e., 0, 1, 2) for specified
          speakers.
        max_files_per_dir (int): Maximum files per directory to load.
        emotions (list): List of emotions we use.

    Attributes:
        labels (numpy.ndarray): List of speaker identifiers determined by
          labelmap. Stored in ``collect_files``.
    """

    def __init__(self, data_root, speakers, labelmap=None, max_files_per_dir=2,
                 emotions=["normal"]):
        for speaker in speakers:
            if speaker not in _speakers:
                raise ValueError(
                    "Unknown speaker '{}'. It should be one of {}".format(
                        speaker, _speakers))

        for emotion in emotions:
            if emotion not in _emotions:
                raise ValueError(
                    "Unknown emotion '{}'. It should be one of {}".format(
                        emotion, _emotions))

        self.data_root = data_root
        self.speakers = speakers
        self.emotions = emotions
        if labelmap is None:
            labelmap = {}
            for idx, speaker in enumerate(speakers):
                labelmap[speaker] = idx
        self.labelmap = labelmap
        self.labels = None
        self.max_files_per_dir = max_files_per_dir

    def collect_files(self):
        """Collect voice statistice wav files
        """
        paths = []
        labels = []
        for speaker in self.speakers:
            dirs = list(map(lambda x: join(self.data_root, _get_dir(speaker, x)),
                            self.emotions))
            files = []
            for d in dirs:
                if not isdir(d):
                    raise RuntimeError("{} doesn't exist.".format(d))

                fs = [join(d, f) for f in listdir(d)]
                fs = list(filter(lambda x: splitext(x)[1] == ".wav", fs))
                fs = sorted(fs)
                fs = fs[:self.max_files_per_dir]
                files.extend(fs)

            for f in files:
                paths.append(f)
                labels.append(self.labelmap[speaker])

        self.labels = np.array(labels, dtype=np.int32)
        return paths
