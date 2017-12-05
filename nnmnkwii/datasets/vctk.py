# coding: utf-8
from __future__ import with_statement, print_function, absolute_import

from nnmnkwii.datasets import FileDataSource

import numpy as np
from os.path import join, splitext, isdir, exists, basename
from glob import glob
from collections import OrderedDict

# List of available speakers.
available_speakers = [
    "225",
    "226",
    "227",
    "228",
    "229",
    "230",
    "231",
    "232",
    "233",
    "234",
    "236",
    "237",
    "238",
    "239",
    "240",
    "241",
    "243",
    "244",
    "245",
    "246",
    "247",
    "248",
    "249",
    "250",
    "251",
    "252",
    "253",
    "254",
    "255",
    "256",
    "257",
    "258",
    "259",
    "260",
    "261",
    "262",
    "263",
    "264",
    "265",
    "266",
    "267",
    "268",
    "269",
    "270",
    "271",
    "272",
    "273",
    "274",
    "275",
    "276",
    "277",
    "278",
    "279",
    "280",
    "281",
    "282",
    "283",
    "284",
    "285",
    "286",
    "287",
    "288",
    "292",
    "293",
    "294",
    "295",
    "297",
    "298",
    "299",
    "300",
    "301",
    "302",
    "303",
    "304",
    "305",
    "306",
    "307",
    "308",
    "310",
    "311",
    "312",
    "313",
    "314",
    # "315", transcriptions are missing, so excludes it here
    "316",
    "317",
    "318",
    "323",
    "326",
    "329",
    "330",
    "333",
    "334",
    "335",
    "336",
    "339",
    "340",
    "341",
    "343",
    "345",
    "347",
    "351",
    "360",
    "361",
    "362",
    "363",
    "364",
    "374",
    "376",
]
assert len(available_speakers) == 108


def _parse_speaker_info(data_root):
    speaker_info_path = join(data_root, "speaker-info.txt")
    if not exists(speaker_info_path):
        raise RuntimeError(
            "speaker-info.txt doesn't exist at \"{}\"".format(speaker_info_path))

    speaker_info = OrderedDict()
    filed_names = ["ID", "AGE", "GENDER", "ACCENTS", "REGION"]
    with open(speaker_info_path, "rb") as f:
        for line in f:
            line = line.decode("utf-8")
            fields = line.split()
            if fields[0] == "ID":
                continue
            assert len(fields) == 4 or len(fields) == 5 or len(fields) == 6
            ID = fields[0]
            speaker_info[ID] = {}
            speaker_info[ID]["AGE"] = int(fields[1])
            speaker_info[ID]["GENDER"] = fields[2]
            speaker_info[ID]["ACCENTS"] = fields[3]
            if len(fields) > 4:
                speaker_info[ID]["REGION"] = " ".join(fields[4:])
            else:
                speaker_info[ID]["REGION"] = ""
    return speaker_info


class _VCTKBaseDataSource(FileDataSource):
    def __init__(self, data_root, speakers, labelmap, max_files):
        self.data_root = data_root
        # accept both e.g., "225" and "p225"
        for idx in range(len(speakers)):
            if speakers[idx][0] == "p":
                speakers[idx] = speakers[idx][1:]
        if speakers == "all":
            speakers = available_speakers
        for speaker in speakers:
            if speaker not in available_speakers:
                raise ValueError(
                    "Unknown speaker '{}'. It should be one of {}".format(
                        speaker, available_speakers))
        self.speakers = speakers
        if labelmap is None:
            labelmap = {}
            for idx, speaker in enumerate(speakers):
                labelmap[speaker] = idx
        self.labelmap = labelmap
        self.labels = None
        self.max_files = max_files

        self.speaker_info = _parse_speaker_info(data_root)
        self._validate()

    def _validate(self):
        # should have pair of transcription and wav files
        for idx, speaker in enumerate(self.speakers):
            txt_files = sorted(glob(join(self.data_root, "txt", "p" + speaker,
                                         "p{}_*.txt".format(speaker))))
            wav_files = sorted(glob(join(self.data_root, "wav48", "p" + speaker,
                                         "p{}_*.wav".format(speaker))))
            assert len(txt_files) > 0
            for txt_path, wav_path in zip(txt_files, wav_files):
                assert splitext(basename(txt_path))[0] == splitext(basename(wav_path))[0]

    def collect_files(self, is_wav):
        if is_wav:
            root = join(self.data_root, "wav48")
            ext = ".wav"
        else:
            root = join(self.data_root, "txt")
            ext = ".txt"

        paths = []
        labels = []

        if self.max_files is None:
            max_files_per_speaker = None
        else:
            max_files_per_speaker = self.max_files // len(self.speakers)
        for idx, speaker in enumerate(self.speakers):
            speaker_dir = join(root, "p" + speaker)
            files = sorted(glob(join(speaker_dir, "p{}_*{}".format(speaker, ext))))
            files = files[:max_files_per_speaker]
            if not is_wav:
                files = list(map(lambda s: open(s, "rb").read().decode("utf-8")[:-1], files))
            for f in files:
                paths.append(f)
                labels.append(self.labelmap[self.speakers[idx]])
        self.labels = np.array(labels, dtype=np.int16)

        return paths


class TranscriptionDataSource(_VCTKBaseDataSource):
    """Transcription data source for VCTK dataset.

    The data source collects text transcriptions from VCTK.
    Users are expected to inherit the class and implement ``collect_features``
    method, which defines how features are computed given a transcription.

    Args:
        data_root (str): Data root.
        speakers (list): List of speakers to find. Speaker id must be ``str``.
          For supported names of speaker, please refer to ``available_speakers``
          defined in the module.
        labelmap (dict[optional]): Dict of speaker labels. If None,
          it's assigned as incrementally (i.e., 0, 1, 2) for specified
          speakers.
        max_files (int): Total number of files to be collected.

    Attributes:
        speaker_info (dict): Dict of speaker information dict. Keyes are speaker
          ids (str) and each value is speaker information consists of ``AGE``,
          ``GENDER`` and ``REGION``.
        labels (numpy.ndarray): Speaker labels paired with collected files.
          Stored in ``collect_files``. This is useful to build multi-speaker
          models.
    """

    def __init__(self, data_root, speakers=available_speakers, labelmap=None, max_files=None):
        super(TranscriptionDataSource, self).__init__(
            data_root, speakers, labelmap, max_files)

    def collect_files(self):
        return super(TranscriptionDataSource, self).collect_files(False)


class WavFileDataSource(_VCTKBaseDataSource):
    """Transcription data source for VCTK dataset.

    The data source collects text transcriptions from VCTK.
    Users are expected to inherit the class and implement ``collect_features``
    method, which defines how features are computed given a transcription.

    Args:
        data_root (str): Data root.
        speakers (list): List of speakers to find. Speaker id must be ``str``.
          For supported names of speaker, please refer to ``available_speakers``
          defined in the module.
        labelmap (dict[optional]): Dict of speaker labels. If None,
          it's assigned as incrementally (i.e., 0, 1, 2) for specified
          speakers.
        max_files (int): Total number of files to be collected.

    Attributes:
        speaker_info (dict): Dict of speaker information dict. Keyes are speaker
          ids (str) and each value is speaker information consists of ``AGE``,
          ``GENDER`` and ``REGION``.
        labels (numpy.ndarray): Speaker labels paired with collected files.
          Stored in ``collect_files``. This is useful to build multi-speaker
          models.
    """

    def __init__(self, data_root, speakers=available_speakers, labelmap=None, max_files=None):
        super(WavFileDataSource, self).__init__(
            data_root, speakers, labelmap, max_files)

    def collect_files(self):
        return super(WavFileDataSource, self).collect_files(True)
