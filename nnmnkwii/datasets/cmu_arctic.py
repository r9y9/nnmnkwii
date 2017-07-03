from __future__ import with_statement, print_function, absolute_import

from os.path import join, expanduser, splitext, isdir
from os import listdir

from nnmnkwii.datasets.decomposed import DecomposedDataset


def _name_to_dirname(name):
    assert len(name) == 3
    return join("cmu_us_{}_arctic".format(name), "wav")


class CMUArctic(DecomposedDataset):
    DATA_ROOT = join(expanduser("~"), "data", "cmu_arctic")

    # Note: idx of the list represents
    speaker_ids = [
        "awb",
        "bdl",
        "clb",
        "jmk",
        "ksp",
        "rms",
        "slt",
    ]

    def __init__(self, *args, **kwargs):
        super(CMUArctic, self).__init__(*args, **kwargs)

    def collect_wav_files(self, speakers):
        speaker_dirs = list(
            map(lambda x: join(self.DATA_ROOT, _name_to_dirname(x)),
                speakers))
        cmu_arctic_all_paths = {}
        for (i, d) in enumerate(speaker_dirs):
            if not isdir(d):
                raise RuntimeError("{} doesn't exist.".format(d))
            files = [join(speaker_dirs[i], f) for f in listdir(d)]
            files = list(filter(lambda x: splitext(x)[1] == ".wav", files))
            cmu_arctic_all_paths[speakers[i]] = sorted(files)
        return cmu_arctic_all_paths
