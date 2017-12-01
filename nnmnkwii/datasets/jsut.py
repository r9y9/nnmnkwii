from __future__ import with_statement, print_function, absolute_import

from nnmnkwii.datasets import FileDataSource

import numpy as np
from os.path import join, splitext, isdir, exists
from os import listdir
from warnings import warn

available_subsets = ["basic5000",
                     "countersuffix26",
                     "loanword128",
                     "onomatopee300",
                     "precedent130",
                     "repeat500",
                     "travel1000",
                     "utparaphrase512",
                     "voiceactress100",
                     ]


class BaseDataSource(FileDataSource):
    def __init__(self, data_root, subset="basic5000"):
        self.subset = subset
        self.data_root = data_root
        transcription_path = join(data_root, subset, "transcript_utf8.txt")
        if not exists(transcription_path):
            raise RuntimeError(
                "transcript_utf8.txt doesn't exists at \"{}\"".format(transcription_path))

        with open(transcription_path, "rb") as f:
            names, transcriptions = [], []
            for line in f:
                line = line.decode("utf-8")
                if ":" not in line:
                    continue
                name, transcription = line.strip().split(":")

                # Hack for jsut_ver1
                if self.subset == "basic5000" and "BASIC4992" in name:
                    name = name.replace("BASIC4992", "BASIC5000")
                elif self.subset == "voiceactress100":
                    if len(name) == len("VOICEACTRESS073"):
                        name = name[:12] + "100_" + name[12:]

                names.append(name)
                transcriptions.append(transcription)
        self.names = np.asarray(names)
        self.transcriptions = np.asarray(transcriptions)

    def validate(self):
        wav_dir = join(self.data_root, self.subset, "wav")
        if not isdir(wav_dir):
            raise RuntimeError("{} doesn't exist.".format(wav_dir))
        miss_indices = []
        for idx, name in enumerate(self.names):
            wav_path = join(wav_dir, name + ".wav")
            if not exists(wav_path):
                miss_indices.append(idx)

        if len(miss_indices) > 0:
            warn("{}/{} wav files were missing in subset {}.".format(
                len(miss_indices), len(self.names), self.subset))

        self.names = np.delete(self.names, miss_indices)
        self.transcriptions = np.delete(self.transcriptions, miss_indices)

    def collect_files(self, is_wav):
        if is_wav:
            wav_dir = join(self.data_root, self.subset, "wav")
            wav_paths = list(map(lambda name: join(wav_dir, name + ".wav"), self.names))
            return np.asarray(wav_paths)
        else:
            return self.transcriptions

    def __len__(self):
        return len(self.names)


class _JSUTFileDataSource(FileDataSource):
    def __init__(self, data_root, subsets, is_wav, validate):
        if subsets == "all":
            subsets = available_subsets
        for subset in subsets:
            if subset not in available_subsets:
                raise ValueError(
                    "Unknown subset '{}'. It should be one of {}".format(
                        subset, available_subsets))

        self.data_root = data_root
        self.subsets = subsets
        self.sub_data_sources = []
        for subset in subsets:
            d = BaseDataSource(data_root, subset)
            if validate:
                d.validate()
            self.sub_data_sources.append(d)
        self.is_wav = is_wav

    def collect_files(self):
        paths = []
        for sub_data_source in self.sub_data_sources:
            paths.extend(sub_data_source.collect_files(self.is_wav))
        return np.asarray(paths)


class TranscriptionDataSource(_JSUTFileDataSource):
    """Transcription data source for JSUT dataset.

    The data source collects text transcriptions from JSUT.
    Users are expected to inherit the class and implement ``collect_features``
    method, which defines how features are computed given a transcription.

    Args:
        data_root (str): Data root.
        subsets (list): Subsets.  Supported names of subset are ``basic5000``,
          ``countersuffix26``, ``loanword128``, ``onomatopee300``,
          ``precedent130``, ``repeat500``, ``travel1000``, ``utparaphrase512``.
          and ``voiceactress100``. Default is ["basic5000"].
    """

    def __init__(self, data_root, subsets=["basic5000"], validate=True):
        super(TranscriptionDataSource, self).__init__(
            data_root, subsets, False, validate)


class WavFileDataSource(_JSUTFileDataSource):
    """Wav file data source for JSUT dataset.

    The data source collects wav files from JSUT.
    Users are expected to inherit the class and implement ``collect_features``
    method, which defines how features are computed given a wav file path.

    Args:
        data_root (str): Data root.
        subsets (list): Subsets.  Supported names of subset are ``basic5000``,
          ``countersuffix26``, ``loanword128``, ``onomatopee300``,
          ``precedent130``, ``repeat500``, ``travel1000``, ``utparaphrase512``.
          and ``voiceactress100``. Default is ["basic5000"].
    """

    def __init__(self, data_root, subsets=["basic5000"], validate=True):
        super(WavFileDataSource, self).__init__(data_root, subsets, True, validate)
