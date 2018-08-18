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

        with open(metadata_path, "rb") as f:
            metadata = []
            for line in f:
                parts = line.decode("utf-8").strip().split("|")
                assert len(parts) == 3
                metadata.append(parts)
        self.metadata = np.asarray(metadata)


class TranscriptionDataSource(LJSpeechDataSource):
    """Transcription data source for LJSpeech dataset.

    The data source collects text transcriptions from LJSpeech.
    Users are expected to inherit the class and implement ``collect_features``
    method, which defines how features are computed given a transcription.

    Args:
        data_root (str): Data root.
        normalized (bool): Collect normalized transcriptions or not.

    Attributes:
        metadata (numpy.ndarray): Metadata, shapeo (``num_files x 3``).
    """

    def __init__(self, data_root, normalized=False):
        super(TranscriptionDataSource, self).__init__(data_root)
        self.normalized = normalized

    def collect_files(self):
        """Collect text transcriptions.

        .. warning::

            Note that it returns list of transcriptions (str), not file paths.

        Returns:
            list: List of text transcription.
        """
        idx = 2 if self.normalized else 1
        return list(self.metadata[:, idx])


class NormalizedTranscriptionDataSource(TranscriptionDataSource):
    """Normalized transcription data source for LJSpeech dataset.

    .. warn::
        Deprecated. Use TranscriptionDataSource with ``normalized=True`` instead.

    Similar to ``LJSpeechTranscriptionDataSource``, but this collect normalized
    transcriptions instead of raw ones.

    Args:
        data_root (str): Data root.

    Attributes:
        metadata (numpy.ndarray): Metadata, shape (``num_files x 3``).
    """

    def __init__(self, data_root):
        super(NormalizedTranscriptionDataSource, self).__init__(data_root, normalized=True)


class WavFileDataSource(LJSpeechDataSource):
    """Wav file data source for LJSpeech dataset.

    The data source collects wav files from LJSpeech.
    Users are expected to inherit the class and implement ``collect_features``
    method, which defines how features are computed given a wav file path.

    Args:
        data_root (str): Data root.

    Attributes:
        metadata (numpy.ndarray): Metadata, shape (``num_files x 3``).
    """

    def __init__(self, data_root):
        super(WavFileDataSource, self).__init__(data_root)

    def collect_files(self):
        """Collect wav files.

        Returns:
            list: List of wav files.
        """
        files = list(map(lambda x: join(self.data_root, "wavs", x + ".wav"),
                         list(self.metadata[:, 0])))
        return files
