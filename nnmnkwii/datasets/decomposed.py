from __future__ import with_statement, print_function, absolute_import

from nnmnkwii.datasets import WavDataset

import pyworld
from scipy.io import wavfile
import numpy as np
from tqdm import tqdm


def _decompose(x, fs, period=5.0):
    """Decompose speech signal into f0, spectral envelope and aperiodicity using WORLD
    """
    f0, timeaxis = pyworld.harvest(x, fs, frame_period=period)
    sp = pyworld.cheaptrick(x, f0, timeaxis, fs)
    ap = pyworld.d4c(x, f0, timeaxis, fs)
    return f0, sp, ap


class DecomposedDataset(WavDataset):
    """Dataset decomposed into acoustic features; f0, spectral envelope and
    aperiodicity, to be built from raw waveforms.
    Label info should also be paired with
    """

    def __init__(self,
                 speakers=None,
                 hop_length=80,
                 limit_ulterances=10,
                 limit_duration_min=None,
                 flatten=False,
                 labelmap=None,
                 parametrize_func=None,
                 silence_threshold=-12,
                 do_build=True,
                 X_f0=None,
                 X_sp=None,
                 X_ap=None,
                 Y=None
                 ):
        self.speakers = speakers

        # wavform analysis parameters
        self.hop_length = hop_length
        self.silence_threshold = silence_threshold
        if parametrize_func is None:
            def __parametrize(f0, sp, ap, fs, period):
                return f0, sp, ap
            self.parametrize_func = __parametrize
        else:
            self.parametrize_func = parametrize_func

        # dataset settings
        self.limit_ulterances = limit_ulterances
        self.limit_duration_min = limit_duration_min
        self.labelmap = labelmap

        # Build dataset
        if do_build:
            self.build(speakers, labelmap)
            self.flattened = False
            if flatten:
                self.flatten()
        else:
            self.X_f0 = X_f0
            self.X_sp = X_ap
            self.X_ap = X_ap

    def __getitem__(self, index):
        return (self.X_f0[index], self.X_sp[index], self.X_ap[index]), self.Y[index]

    def __len__(self):
        return self.X_f0.shape[0]

    def __add__(self, other):
        d = type(self)(do_build=False)
        assert self.labelmap is not None
        assert other.labelmap is not None
        d.__dict__.update(self.__dict__)
        d.X_f0 = np.vstack((d.X_f0, other.X_f0))
        d.X_sp = np.vstack((d.X_sp, other.X_sp))
        d.X_ap = np.vstack((d.X_sp, other.X_ap))
        d.Y = np.vstack((d.Y, other.Y))
        d.paths.update(other.paths)
        d.labelmap.update(other.labelmap)
        d.speakers = self.speakers + other.speakers
        return d

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
                fs, x = wavfile.read(f)
                frame_period = self.hop_length / fs * 1000
                total_seconds += len(x) / fs
                # need to convert float64 array to perform WORLD analysis on it
                x = x.astype(np.float64)
                f0, sp, ap = _decompose(x, fs, frame_period)

                # apply parametrize function
                f0_param, sp_param, ap_param = self.parametrize_func(
                    f0, sp, ap, fs, frame_period)

                # Force dtype
                f0_param = f0_param.astype(np.float32)[:, None]
                sp_param = sp_param.astype(np.float32)
                ap_param = ap_param.astype(np.float32)

                X_f0.append(f0_param)
                X_sp.append(sp_param)
                X_ap.append(ap_param)
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
        Y = np.array(list(self.Y[0]) * len(self.X_f0[0])
                     )[:, None].astype(np.float32)
        for i in range(1, len(self.X_f0)):
            X_f0 = np.vstack((X_f0, self.X_f0[i]))
            X_sp = np.vstack((X_sp, self.X_sp[i]))
            X_ap = np.vstack((X_ap, self.X_ap[i]))
            Y = np.vstack(
                (Y, np.array(list(self.Y[i]) * len(self.X_f0[i]))[:, None].astype(np.float32)))
        assert X_f0.dtype == "f" and Y.dtype == "f"
        self.X_f0, self.X_sp, self.X_ap, self.Y = X_f0, X_sp, X_ap, Y
        self.flattened = True
