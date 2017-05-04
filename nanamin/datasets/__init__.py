from __future__ import with_statement, print_function, absolute_import

import scipy.io.wavfile
import pyworld
import librosa
import numpy as np
from tqdm import tqdm


class Dataset(object):
    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class WavDataset(object):
    def collect_wav_files(self, speakers):
        """Each dataset inherits this class must provides a way to collect
        wav file paths for specified speaker(s)
        """
        raise NotImplementedError


class DecomposedDataset(WavDataset):
    """Dataset decomposed into acoustic features; f0, spectral envelope and
    aperiodicity, to be built from raw waveforms.
    Label info should also be paired with
    """

    def __init__(self,
                 speakers=None,
                 hop_length=80,
                 limit_ulterances=None,
                 limit_duration_min=10,
                 enable_trim=True,
                 flatten=True,
                 labelmap=None
                 ):
        assert speakers is not None
        self.speakers = speakers

        # wavform analysis parameters
        self.hop_length = hop_length

        # dataset settings
        self.limit_ulterances = limit_ulterances
        self.limit_duration_min = limit_duration_min
        self.enable_trim = enable_trim

        # Build dataset
        self.build(speakers, labelmap)
        self.flattened = False
        if flatten:
            self.flatten()

    def __getitem__(self, index):
        return (self.X_f0[index], self.X_sp[index], self.X_ap[index]), self.Y[index]

    def __len__(self):
        return self.X_f0.shape[0]

    def build(self, speakers, labelmap=None):
        paths = self.collect_wav_files(speakers)
        self.paths = paths
        if labelmap is None:
            labelmap = {f: idx for (idx, f) in enumerate(speakers)}

        X_f0 = []
        X_sp = []
        X_ap = []
        Y = []

        # for each active speaker
        for speaker_name in speakers:
            total_seconds = 0
            speaker_paths = paths[speaker_name]
            print("Collecting speaker features for {}".format(speaker_name))

            # for each ulterance
            for (i, f) in tqdm(enumerate(speaker_paths[:self.limit_ulterances])):
                # Read audio file and then process it with WORLD
                fs, x = scipy.io.wavfile.read(f)
                frame_period_msec = self.hop_length / fs * 1000
                total_seconds += len(x) / fs
                # need to convert float64 array to perform WORLD analysis on it
                x = x.astype(np.float64)
                if self.enable_trim:
                    x, _ = librosa.effects.trim(
                        x, top_db=50, frame_length=1024, hop_length=self.hop_length)
                    f0, sp, ap = pyworld.wav2world(
                        x, fs, frame_period=frame_period_msec)

                f0 = f0.astype(np.float32)[:, None]
                logsp = np.log(sp).astype(np.float32)
                ap = ap.astype(np.float32)

                X_f0.append(f0)
                X_sp.append(logsp)
                X_ap.append(ap)
                Y.append([labelmap[speaker_name]])

                if self.limit_duration_min is not None and \
                        total_seconds > 60 * self.limit_duration_min:
                    print("Collected duration ({} mins) exceeds limit length, break".format(
                        total_seconds / 60))
                    break

        self.X_f0 = np.array(X_f0)
        self.X_sp = np.array(X_sp)
        self.X_ap = np.array(X_ap)
        self.Y = np.array(Y).astype(np.float32)
        assert self.X_f0.dtype == "O"

    def flatten(self):
        """make all ulterances into a single array
        """
        if self.flattened:
            return

        X_f0 = self.X_f0[0]
        X_sp = self.X_sp[0]
        X_f0 = self.X_f0[0]

        X_f0 = self.X_f0[0]
        X_sp = self.X_sp[0]
        X_ap = self.X_ap[0]
        Y = np.array(list(self.Y[0]) * len(self.X_f0[0]))[:, None]
        for i in range(1, len(self.X_f0)):
            X_f0 = np.vstack((X_f0, self.X_f0[i]))
            X_sp = np.vstack((X_sp, self.X_sp[i]))
            X_ap = np.vstack((X_ap, self.X_ap[i]))
            Y = np.vstack(
                (Y, np.array(list(self.Y[i]) * len(self.X_f0[i]))[:, None]))
        assert X_f0.dtype == "f" and Y.dtype == "f"
        self.X_f0, self.X_sp, self.X_ap, self.Y = X_f0, X_sp, X_ap, Y
        self.flattened = True
