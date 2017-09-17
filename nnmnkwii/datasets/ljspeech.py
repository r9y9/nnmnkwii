from __future__ import with_statement, print_function, absolute_import

from nnmnkwii.datasets import FileDataSource

from os.path import join, exists
import numpy as np


class LJSpeechDataSource(FileDataSource):
    def __init__(self, data_root):
        self.data_root = data_root
        metadata_path = join(data_root, "metadata.csv")
        if not exists(metadata_path):
            raise RuntimeError(
                "metadata.csv doesn't exists at \"{}\"".format(metadata_path))

        with open(metadata_path, encoding="utf-8") as f:
            metadata = []
            for line in f:
                parts = line.strip().split("|")
                assert len(parts) == 3
                metadata.append(parts)
        self.metadata = np.asarray(metadata)


class LJSpeechTranscriptionDataSource(LJSpeechDataSource):
    """Transcription data source for LJSpeech dataset.

    A builtin implementation of ``FileDataSource`` for LJSpeech transcriptions.
    Users are expected to inherit the class and implement ``collect_features``
    method, which defines how features are computed given a transcription.

    Args:
        data_root (str): Data root.

    Attributes:
        metadata (numpy.ndarray) Metadata, shapeo (``num_files x 3``).
    """

    def __init__(self, data_root):
        super(LJSpeechTranscriptionDataSource, self).__init__(data_root)

    def collect_files(self):
        return list(self.metadata[:, 1])


class LJSpeechNormalizedTranscriptionDataSource(LJSpeechDataSource):
    """Normalized transcription data source for LJSpeech dataset.

    Similar to ``LJSpeechTranscriptionDataSource``, but this collect normalized
    transcriptions instead of raw ones.

    Args:
        data_root (str): Data root.

    Attributes:
        metadata (numpy.ndarray) Metadata, shapeo (``num_files x 3``).
    """

    def __init__(self, data_root):
        super(LJSpeechNormalizedTranscriptionDataSource,
              self).__init__(data_root)

    def collect_files(self):
        return list(self.metadata[:, 2])


class LJSpeechWavFileDataSource(LJSpeechDataSource):
    """Wav file data source for LJSpeech dataset.

    A builtin implementation of ``FileDataSource`` for LJSpeech wav files.
    Users are expected to inherit the class and implement ``collect_features``
    method, which defines how features are computed given a wav path.

    Args:
        data_root (str): Data root.

    Attributes:
        metadata (numpy.ndarray) Metadata, shapeo (``num_files x 3``).
    """

    def __init__(self, data_root):
        super(LJSpeechWavFileDataSource, self).__init__(data_root)

    def collect_files(self):
        files = list(map(lambda x: join(self.data_root, "wavs", x + ".wav"),
                         list(self.metadata[:, 0])))
        return files
