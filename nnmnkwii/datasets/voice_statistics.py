from __future__ import with_statement, print_function, absolute_import

from nnmnkwii.datasets import FileDataSource

import numpy as np
from os.path import join, splitext, isdir, exists
from os import listdir

available_speakers = ["fujitou", "tsuchiya", "uemura", "hiroshiba"]
available_emotions = ["angry", "happy", "normal"]


def _get_dir(speaker, emotion):
    return "{}_{}".format(speaker, emotion)


class TranscriptionDataSource(FileDataSource):
    """Transcription data source for VoiceStatistics dataset

    Users are expected to inherit the class and implement ``collect_features``
    method, which defines how features are computed given a transcription.

    Args:
        data_root (str): Data root.
        column (str): ``sentense``, ``yomi`` or ``monophone``.
        max_files (int): Total number of files to be collected.

    Atributes:
        transcriptions (list): Transcriptions.
    """
    column_map = {"sentence_id": 0, "sentence": 1, "yomi": 2, "monophone": 3}

    def __init__(self, data_root, column="sentence", max_files=None):
        path = join(data_root, "balance_sentences.txt")
        if not exists(path):
            raise RuntimeError(
                "balance_sentences.txt doesn't exist at \"{}\"".format(path))

        self.transcriptions = []
        self.max_files = max_files
        if column not in self.column_map:
            raise ValueError(
                "Not supported column {}. It should be one of 'sentense', 'yomi' or 'monophone'.".format(column))
        with open(path) as f:
            for l in f:
                # header
                if l.startswith("sentence_id"):
                    continue
                v = l.split("\t")[self.column_map[column]].strip()
                self.transcriptions.append(v)
        assert len(self.transcriptions) == 100

    def collect_files(self):
        """Collect text transcriptions.

        .. warning::

            Note that it returns list of transcriptions (str), not file paths.

        Returns:
            list: List of text transcription.
        """
        if self.max_files is None:
            return self.transcriptions
        else:
            return self.transcriptions[:self.max_files]


class WavFileDataSource(FileDataSource):
    """Wav file data source for Voice-statistics dataset.

    The data source collects wav files from voice-statistics.
    Users are expected to inherit the class and implement
    ``collect_features`` method, which defines how features are computed
    given a wav file path.

    Args:
        data_root (str): Data root
        speakers (list): List of speakers to load. Supported names of speaker
          are ``fujitou``, ``tsuchiya`` and ``uemura``.
        labelmap (dict[optional]): Dict of speaker labels. If None,
          it's assigned as incrementally (i.e., 0, 1, 2) for specified
          speakers.
        max_files (int): Total number of files to be collected.
        emotions (list): List of emotions we use. Supported names of emotions
          are ``angry``, ``happy`` and ``normal``.

    Attributes:
        labels (numpy.ndarray): List of speaker identifiers determined by
          labelmap. Stored in ``collect_files``.
    """

    def __init__(self, data_root, speakers, labelmap=None, max_files=None,
                 emotions=["normal"]):
        for speaker in speakers:
            if speaker not in available_speakers:
                raise ValueError(
                    "Unknown speaker '{}'. It should be one of {}".format(
                        speaker, available_speakers))

        for emotion in emotions:
            if emotion not in available_emotions:
                raise ValueError(
                    "Unknown emotion '{}'. It should be one of {}".format(
                        emotion, available_emotions))

        self.data_root = data_root
        self.speakers = speakers
        self.emotions = emotions
        if labelmap is None:
            labelmap = {}
            for idx, speaker in enumerate(speakers):
                labelmap[speaker] = idx
        self.labelmap = labelmap
        self.labels = None
        self.max_files = max_files

    def collect_files(self):
        """Collect wav files for specific speakers.

        Returns:
            list: List of collected wav files.
        """
        paths = []
        labels = []

        if self.max_files is None:
            max_files_per_dir = None
        else:
            max_files_per_dir = self.max_files // len(self.emotions) \
                // len(self.speakers)
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
                fs = fs[:max_files_per_dir]
                files.extend(fs)

            for f in files:
                paths.append(f)
                labels.append(self.labelmap[speaker])

        self.labels = np.array(labels, dtype=np.int32)
        return paths


# For compat, remove this after v0.1.0
VoiceStatisticsWavFileDataSource = WavFileDataSource
