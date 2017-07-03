from __future__ import with_statement, print_function, absolute_import


class Dataset(object):
    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class WavDataset(Dataset):
    def collect_wav_files(self, speakers):
        """Each dataset inherits this class must provides a way to collect
        wav file paths for specified speaker(s)
        """
        raise NotImplementedError
