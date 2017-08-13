from __future__ import with_statement, print_function, absolute_import

from nnmnkwii.datasets import FileDataSource

from os.path import join, splitext
from os import listdir

_speakers = ["fujitou", "tsuchiya", "uemura"]
_emotions = ["angry", "happy", "normal"]


def _get_dir(speaker, emotion):
    return "{}_{}".format(speaker, emotion)


class VoiceStatisticsWavFileDataSource(FileDataSource):
    """Voice-statistics data source

    You can get the dataset from `voice-statistics`_.

    _voice-statistics: http://voice-statistics.github.io/

    Attributes:
        data_root: Data root
        speakers (list): List of speakers to load.
        labelmap (dict[optional]): Dict of speaker labels. If None,
            it's assigned as incrementally (i.e., 0, 1, 2) for specified
            speakers.
        max_files_per_dir (int): Maximum files per directory to load.
        emotions (list): List of emotions we use.
    """

    def __init__(self, data_root, speakers, labelmap=None, max_files_per_dir=2,
                 emotions=["normal"]):
        for speaker in speakers:
            if not speaker in _speakers:
                raise RuntimeError(
                    "Unknown speaker '{}'. It should be one of {}".format(
                        speaker, _speakers))

        for emotion in emotions:
            if not emotion in _emotions:
                raise RuntimeError(
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
                fs = [join(d, f) for f in listdir(d)]
                fs = list(filter(lambda x: splitext(x)[1] == ".wav", fs))
                fs = sorted(fs)
                fs = fs[:self.max_files_per_dir]
                files.extend(fs)
            files = sorted(files)

            for f in files[:self.max_files_per_dir]:
                paths.append(f)
                labels.append(self.labelmap[speaker])

        self.labels = labels
        return paths
