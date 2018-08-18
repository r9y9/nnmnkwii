# coding: utf-8
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
from nnmnkwii.datasets import cmu_arctic, voice_statistics, ljspeech, vcc2016
from nnmnkwii.datasets import jsut, vctk

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

    @raises(RuntimeError)
    def __test_no_trans():
        voice_statistics.TranscriptionDataSource("dummy")

    __test_invalid_speaker()
    __test_invalid_emotion()
    __test_nodir(data_source)
    __test_no_trans()


def test_ljspeech_dummy():
    data_sources = [ljspeech.TranscriptionDataSource,
                    ljspeech.NormalizedTranscriptionDataSource,
                    ljspeech.WavFileDataSource]

    for data_source in data_sources:
        @raises(RuntimeError)
        def f(source):
            source("dummy")

        f(data_source)


def test_vcc2016_dummy():
    data_source = vcc2016.WavFileDataSource("dummy", speakers=["SF1"])

    @raises(ValueError)
    def __test_invalid_speaker():
        data_source = vcc2016.WavFileDataSource("dummy", speakers=["test"])

    @raises(RuntimeError)
    def __test_nodir(data_source):
        data_source.collect_files()

    __test_invalid_speaker()
    __test_nodir(data_source)


def test_jsut_dummy():
    data_sources = [jsut.TranscriptionDataSource,
                    jsut.WavFileDataSource]

    for data_source in data_sources:
        @raises(RuntimeError)
        def f(source):
            source("dummy")

        f(data_source)


def test_vctk_dummy():
    assert len(vctk.available_speakers) == 108
    data_sources = [vctk.TranscriptionDataSource,
                    vctk.WavFileDataSource]

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

    # Use all data
    data_source = MyFileDataSource(
        DATA_DIR, speakers=["clb", "slt"], max_files=None)
    X = FileSourceDataset(data_source)
    assert len(X) == 1132 * 2


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

    # Use all data
    data_source = MyFileDataSource(
        DATA_DIR, speakers=["fujitou", "tsuchiya"], max_files=None)
    X = FileSourceDataset(data_source)
    assert len(X) == 100 * 2

    # Transcriptions
    source = voice_statistics.TranscriptionDataSource(DATA_DIR)
    texts = source.collect_files()
    assert len(texts) == 100
    assert texts[0] == "また東寺のように五大明王と呼ばれる主要な明王の中央に配されることも多い"

    source = voice_statistics.TranscriptionDataSource(DATA_DIR, column="yomi")
    texts = source.collect_files()
    assert len(texts) == 100
    assert texts[0] == "マタトージノヨーニゴダイミョウオートヨバレルシュヨーナミョーオーノチューオーニハイサレルコトモオーイ"

    source = voice_statistics.TranscriptionDataSource(DATA_DIR, column="monophone")
    texts = source.collect_files()
    assert len(texts) == 100
    s = "s/u,m/a:,t/o,f/o,N,k/a,r/a,f/i:,ch/a:,f/o,N,m/a,d/e,m/a,r/u,ch/i,d/e,b/a,i,s/u,n/i,t/a,i,o:"
    assert texts[9] == s

    source = voice_statistics.TranscriptionDataSource(DATA_DIR, max_files=10)
    texts = source.collect_files()
    assert len(texts) == 10
    assert texts[0] == "また東寺のように五大明王と呼ばれる主要な明王の中央に配されることも多い"


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

    class MyNormalizedTextDataSource(ljspeech.TranscriptionDataSource):
        def __init__(self, data_root):
            super(MyNormalizedTextDataSource, self).__init__(data_root, normalized=True)

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


@attr("require_local_data")
@attr("require_vcc2016")
def test_vcc2016():
    DATA_DIR = join(expanduser("~"), "data", "vcc2016")
    if not exists(DATA_DIR):
        warn("Data doesn't exist at {}".format(DATA_DIR))
        return

    class MyFileDataSource(vcc2016.WavFileDataSource):
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
        DATA_DIR, speakers=["SF1"], max_files=max_files)
    X = FileSourceDataset(data_source)
    assert len(X) == max_files
    print(X[0].shape)  # warmup collect_features path

    # Multi speakers
    data_source = MyFileDataSource(
        DATA_DIR, speakers=["SF1", "SF2"], max_files=max_files)
    X = FileSourceDataset(data_source)
    assert len(X) == max_files

    # Speaker labels
    Y = data_source.labels
    assert np.all(Y[:max_files // 2] == 0)
    assert np.all(Y[max_files // 2:] == 1)

    # Custum speaker id
    data_source = MyFileDataSource(
        DATA_DIR, speakers=["SF1", "SF2"], max_files=max_files,
        labelmap={"SF1": 1, "SF2": 0})
    X = FileSourceDataset(data_source)
    Y = data_source.labels
    assert np.all(Y[:max_files // 2] == 1)
    assert np.all(Y[max_files // 2:] == 0)

    # Use all data
    data_source = MyFileDataSource(
        DATA_DIR, speakers=["SF1", "SF2"], max_files=None)
    X = FileSourceDataset(data_source)
    assert len(X) == 162 * 2


@attr("require_local_data")
@attr("require_jsut")
def test_jsut():
    DATA_DIR = join(expanduser("~"), "data", "jsut_ver1.1")
    if not exists(DATA_DIR):
        warn("Data doesn't exist at {}".format(DATA_DIR))
        return

    class MyTextDataSource(jsut.TranscriptionDataSource):
        def __init__(self, data_root, subsets):
            super(MyTextDataSource, self).__init__(data_root, subsets)

        def collect_features(self, text):
            return text

    data_source = MyTextDataSource(DATA_DIR, subsets=["basic5000"])
    X1 = FileSourceDataset(data_source)
    assert X1[0] == u"水をマレーシアから買わなくてはならないのです。"

    data_source = MyTextDataSource(DATA_DIR, subsets=["travel1000"])
    X2 = FileSourceDataset(data_source)
    assert X2[0] == u"あなたの荷物は、ロサンゼルスに残っています。"

    # Multiple subsets
    data_source = MyTextDataSource(DATA_DIR, subsets=["basic5000", "travel1000"])
    X3 = FileSourceDataset(data_source)
    assert X3[0] == u"水をマレーシアから買わなくてはならないのです。"
    assert len(X3) == len(X1) + len(X2)

    # All subsets
    data_source = MyTextDataSource(DATA_DIR, subsets=jsut.available_subsets)
    X = FileSourceDataset(data_source)
    # As of 2017/11/2. There were 30 missing wav files.
    # This should be 7696
    assert len(X) == 7696

    class MyWavFileDataSource(jsut.WavFileDataSource):
        def __init__(self, data_root, subsets):
            super(MyWavFileDataSource, self).__init__(data_root, subsets)
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

    data_source = MyWavFileDataSource(DATA_DIR, subsets=["basic5000"])
    X = FileSourceDataset(data_source)
    print(X[0].shape)


@attr("require_local_data")
@attr("require_vctk")
def test_vctk():
    DATA_DIR = join(expanduser("~"), "data", "VCTK-Corpus")
    if not exists(DATA_DIR):
        warn("Data doesn't exist at {}".format(DATA_DIR))
        return

    class MyTextDataSource(vctk.TranscriptionDataSource):
        def __init__(self, data_root, speakers, labelmap=None, max_files=None):
            super(MyTextDataSource, self).__init__(data_root, speakers, labelmap, max_files)

        def collect_features(self, text):
            return text

    # Single speaker
    data_source = MyTextDataSource(DATA_DIR, speakers=["225"])
    X = FileSourceDataset(data_source)
    assert X[0] == "Please call Stella."
    n_225 = len(X)

    data_source = MyTextDataSource(DATA_DIR, speakers=["p228"])
    X = FileSourceDataset(data_source)
    assert X[0] == "Please call Stella."
    n_228 = len(X)

    # multiple speakers
    data_source = MyTextDataSource(DATA_DIR, speakers=["225", "228"])
    X = FileSourceDataset(data_source)
    assert len(X) == n_225 + n_228

    # All speakers
    data_source = MyTextDataSource(DATA_DIR, speakers=vctk.available_speakers)
    X = FileSourceDataset(data_source)
    assert X[0] == "Please call Stella."
    assert len(X) == 44085

    # Speaker labels
    data_source = MyTextDataSource(DATA_DIR, speakers=["225", "228"])
    X = FileSourceDataset(data_source)
    labels = data_source.labels
    assert len(X) == len(labels)
    assert (labels[:n_225] == 0).all()
    assert (labels[n_225:] == 1).all()

    # max files
    max_files = 16
    data_source = MyTextDataSource(DATA_DIR, speakers=["225", "228"], max_files=max_files)
    X = FileSourceDataset(data_source)
    assert len(X) == max_files
    Y = data_source.labels
    assert np.all(Y[:max_files // 2] == 0)
    assert np.all(Y[max_files // 2:] == 1)

    # Custum labelmap
    data_source = MyTextDataSource(DATA_DIR, speakers=["225", "228"],
                                   labelmap={"225": 225, "228": 228})
    X = FileSourceDataset(data_source)
    labels = data_source.labels
    assert len(X) == len(labels)
    assert (labels[:n_225] == 225).all()
    assert (labels[n_225:] == 228).all()

    class MyWavFileDataSource(vctk.WavFileDataSource):
        def __init__(self, data_root, speakers, labelmap=None):
            super(MyWavFileDataSource, self).__init__(data_root, speakers, labelmap)
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

    data_source = MyWavFileDataSource(DATA_DIR, speakers=["225"])
    X = FileSourceDataset(data_source)
    print(X[0].shape)
