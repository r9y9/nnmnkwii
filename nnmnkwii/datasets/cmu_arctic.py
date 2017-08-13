from __future__ import with_statement, print_function, absolute_import

from nnmnkwii.datasets import FileDataSource

from os.path import join, splitext, isdir
from os import listdir


def _name_to_dirname(name):
    assert len(name) == 3
    return join("cmu_us_{}_arctic".format(name), "wav")


class CMUArcticWavFileDataSource(FileDataSource):
    def __init__(self, data_root, speakers, labelmap=None, max_files=2):
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

        self.labels = labels
        return paths
