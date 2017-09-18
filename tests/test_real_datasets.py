from __future__ import division, print_function, absolute_import

from nnmnkwii.datasets import FileSourceDataset, PaddedFileSourceDataset

import numpy as np
from nose.tools import raises
from nose.plugins.attrib import attr
from warnings import warn

from os.path import join, dirname, expanduser, exists
from scipy.io import wavfile
import pysptk
import pyworld
from nnmnkwii.preprocessing import trim_zeros_frames

# Data source implementations
from nnmnkwii.datasets import cmu_arctic, voice_statistics, ljspeech

# Tests marked with "require_local_data" needs data to be downloaded.


def test_cmu_arctic_dummy():
    data_source = cmu_arctic.WavFileDataSource("dummy", speakers=["clb"])

    @raises(ValueError)
    def __test_invalid_speaker():
        data_source = cmu_arctic.WavFileDataSource("dummy", speakers=["test"])

    @raises(RuntimeError)
    def __test_nodir(data_source):
        data_source.collect_files()

    __test_invalid_speaker()
    __test_nodir(data_source)


def test_voice_statistics_dummy():
    data_source = voice_statistics.WavFileDataSource("dummy", speakers=["fujitou"])

    @raises(ValueError)
    def __test_invalid_speaker():
        data_source = voice_statistics.WavFileDataSource("dummy", speakers=["test"])

    @raises(ValueError)
    def __test_invalid_emotion():
        data_source = voice_statistics.WavFileDataSource(
            "dummy", speakers=["fujitou"], emotions="nnmnkwii")

    @raises(RuntimeError)
    def __test_nodir(data_source):
        data_source.collect_files()

    __test_invalid_speaker()
    __test_invalid_emotion()
    __test_nodir(data_source)


def test_ljspeech_dummy():
    data_sources = [ljspeech.TranscriptionDataSource,
                    ljspeech.NormalizedTranscriptionDataSource,
                    ljspeech.WavFileDataSource]

    for data_source in data_sources:
        @raises(RuntimeError)
        def f(source):
            source("dummy")

        f(data_source)


@attr("require_local_data")
@attr("require_cmu_arctic")
def test_cmu_arctic():
    DATA_DIR = join(expanduser("~"), "data", "cmu_arctic")
    if not exists(DATA_DIR):
        warn("Data doesn't exist at {}".format(DATA_DIR))
        return

    class MyFileDataSource(cmu_arctic.WavFileDataSource):
        def __init__(self, data_root, speakers, labelmap=None, max_files=2):
            super(MyFileDataSource, self).__init__(
                data_root, speakers, labelmap=labelmap, max_files=max_files)
            self.alpha = pysptk.util.mcepalpha(16000)

        def collect_features(self, path):
            fs, x = wavfile.read(path)
            x = x.astype(np.float64)
            f0, timeaxis = pyworld.dio(x, fs, frame_period=5)
            f0 = pyworld.stonemask(x, f0, timeaxis, fs)
            spectrogram = pyworld.cheaptrick(x, f0, timeaxis, fs)
            spectrogram = trim_zeros_frames(spectrogram)
            mc = pysptk.sp2mc(spectrogram, order=24, alpha=self.alpha)
            return mc.astype(np.float32)

    max_files = 10
    data_source = MyFileDataSource(
        DATA_DIR, speakers=["clb"], max_files=max_files)
    X = FileSourceDataset(data_source)
    assert len(X) == max_files
    print(X[0].shape)  # warmup collect_features path

    # Multi speakers
    data_source = MyFileDataSource(
        DATA_DIR, speakers=["clb", "slt"], max_files=max_files)
    X = FileSourceDataset(data_source)
    assert len(X) == max_files

    # Speaker labels
    Y = data_source.labels
    assert np.all(Y[:max_files // 2] == 0)
    assert np.all(Y[max_files // 2:] == 1)

    # Custum speaker id
    data_source = MyFileDataSource(
        DATA_DIR, speakers=["clb", "slt"], max_files=max_files,
        labelmap={"clb": 1, "slt": 0})
    X = FileSourceDataset(data_source)
    Y = data_source.labels
    assert np.all(Y[:max_files // 2] == 1)
    assert np.all(Y[max_files // 2:] == 0)


@attr("require_local_data")
@attr("require_voice_statistics")
def test_voice_statistics():
    DATA_DIR = join(expanduser("~"), "data", "voice-statistics")
    if not exists(DATA_DIR):
        warn("Data doesn't exist at {}".format(DATA_DIR))
        return

    class MyFileDataSource(voice_statistics.WavFileDataSource):
        def __init__(self, data_root, speakers, emotions=["normal"],
                     labelmap=None, max_files=2):
            super(MyFileDataSource, self).__init__(
                data_root, speakers, emotions=emotions, labelmap=labelmap,
                max_files=max_files)
            self.alpha = pysptk.util.mcepalpha(48000)

        def collect_features(self, path):
            fs, x = wavfile.read(path)
            assert fs == 48000
            x = x.astype(np.float64)
            f0, timeaxis = pyworld.dio(x, fs, frame_period=5)
            f0 = pyworld.stonemask(x, f0, timeaxis, fs)
            spectrogram = pyworld.cheaptrick(x, f0, timeaxis, fs)
            spectrogram = trim_zeros_frames(spectrogram)
            mc = pysptk.sp2mc(spectrogram, order=24, alpha=self.alpha)
            return mc.astype(np.float32)

    max_files = 40
    data_source = MyFileDataSource(
        DATA_DIR, speakers=["fujitou"], max_files=max_files)
    X = FileSourceDataset(data_source)
    Y = data_source.labels
    print(len(X), max_files)
    assert len(X) == max_files
    assert np.all(Y == 0)
    print(X[0].shape)  # warmup collect_features path]

    # Multi speakers
    data_source = MyFileDataSource(
        DATA_DIR, speakers=["fujitou", "tsuchiya"],
        max_files=max_files)
    X = FileSourceDataset(data_source)
    Y = data_source.labels
    assert len(X) == max_files
    assert np.all(Y[:max_files // 2] == 0)
    assert np.all(Y[max_files // 2:] == 1)

    # Multi speakers + Multi emotions
    data_source = MyFileDataSource(
        DATA_DIR, speakers=["fujitou", "tsuchiya"], emotions=["normal", "happy"],
        max_files=max_files)
    X = FileSourceDataset(data_source)
    Y = data_source.labels
    assert len(X) == max_files
    assert np.all(Y[:max_files // 2] == 0)
    assert np.all(Y[max_files // 2:] == 1)


@attr("require_local_data")
@attr("require_ljspeech")
def test_ljspeech():
    DATA_DIR = join(expanduser("~"), "data", "LJSpeech-1.0")
    if not exists(DATA_DIR):
        warn("Data doesn't exist at {}".format(DATA_DIR))
        return

    class MyTextDataSource(ljspeech.TranscriptionDataSource):
        def __init__(self, data_root):
            super(MyTextDataSource, self).__init__(data_root)

        def collect_features(self, text):
            return text

    class MyNormalizedTextDataSource(ljspeech.NormalizedTranscriptionDataSource):
        def __init__(self, data_root):
            super(MyNormalizedTextDataSource, self).__init__(data_root)

        def collect_features(self, text):
            return text

    data_source = MyTextDataSource(DATA_DIR)
    X = FileSourceDataset(data_source)
    assert X[1] == "in being comparatively modern."

    data_source = MyNormalizedTextDataSource(DATA_DIR)
    X = FileSourceDataset(data_source)
    assert X[1] == "in being comparatively modern."

    class MyWavFileDataSource(ljspeech.WavFileDataSource):
        def __init__(self, data_root):
            super(MyWavFileDataSource, self).__init__(data_root)
            self.alpha = pysptk.util.mcepalpha(22050)

        def collect_features(self, path):
            fs, x = wavfile.read(path)
            assert fs == 22050
            x = x.astype(np.float64)
            f0, timeaxis = pyworld.dio(x, fs, frame_period=5)
            f0 = pyworld.stonemask(x, f0, timeaxis, fs)
            spectrogram = pyworld.cheaptrick(x, f0, timeaxis, fs)
            spectrogram = trim_zeros_frames(spectrogram)
            mc = pysptk.sp2mc(spectrogram, order=24, alpha=self.alpha)
            return mc.astype(np.float32)

    data_source = MyWavFileDataSource(DATA_DIR)
    X = FileSourceDataset(data_source)
    print(X[0].shape)
