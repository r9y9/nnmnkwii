from __future__ import division, print_function, absolute_import

from nnmnkwii.datasets import FileSourceDataset, PaddedFileSourceDataset
from nnmnkwii.datasets import MemoryCacheFramewiseDataset
from nnmnkwii.datasets import MemoryCacheDataset
from nnmnkwii.util import example_file_data_sources_for_acoustic_model
from nnmnkwii.util import example_file_data_sources_for_duration_model

import torch
from torch.utils import data as data_utils
from torch.autograd import Variable
from torch import nn
from torch import optim

import numpy as np
from nose.tools import raises
from os.path import join, dirname

DATA_DIR = join(dirname(__file__), "data")


def _get_small_datasets(padded=False, duration=False):
    if duration:
        X, Y = example_file_data_sources_for_duration_model()
    else:
        X, Y = example_file_data_sources_for_acoustic_model()
    if padded:
        X = PaddedFileSourceDataset(X, padded_length=1000)
        Y = PaddedFileSourceDataset(Y, padded_length=1000)
    else:
        X = FileSourceDataset(X)
        Y = FileSourceDataset(Y)
    return X, Y


class PyTorchDataset(data_utils.Dataset):
    def __init__(self, X, Y, lengths):
        self.X = X
        self.Y = Y
        self.lengths = lengths

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx])
        y = torch.from_numpy(self.Y[idx])
        l = torch.from_numpy(self.lengths[idx])
        return x, y, l

    def __len__(self):
        return len(self.X)


class MyRNN(nn.Module):
    def __init__(self, D_in, H, D_out, num_layers=1, bidirectional=True):
        super(MyRNN, self).__init__()
        self.hidden_dim = H
        self.num_layers = num_layers
        self.num_direction = 2 if bidirectional else 1
        self.lstm = nn.LSTM(D_in, H, num_layers, bidirectional=bidirectional,
                            batch_first=True)
        self.hidden2out = nn.Linear(
            self.num_direction * self.hidden_dim, D_out)

    def init_hidden(self, batch_size):
        h, c = (Variable(torch.zeros(self.num_layers * self.num_direction,
                                     batch_size, self.hidden_dim)),
                Variable(torch.zeros(self.num_layers * self.num_direction,
                                     batch_size, self.hidden_dim)))
        return h, c

    def forward(self, sequence, lengths, h, c):
        sequence = nn.utils.rnn.pack_padded_sequence(sequence, lengths,
                                                     batch_first=True)
        output, (h, c) = self.lstm(sequence, (h, c))
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(
            output, batch_first=True)
        output = self.hidden2out(output)
        return output


def test_pack_sequnce():
    """Test minibatch RNN training using pack_pad_sequence.
    """

    X, Y = _get_small_datasets(padded=False)
    lengths = np.array([len(x) for x in X], dtype=np.int)[:, None]

    # We need padded dataset
    X, Y = _get_small_datasets(padded=True)

    # For the above reason, we need to explicitly give the number of frames.
    X = MemoryCacheDataset(X, cache_size=len(X))
    Y = MemoryCacheDataset(Y, cache_size=len(Y))

    in_dim = X[0].shape[-1]
    out_dim = Y[0].shape[-1]
    hidden_dim = 5
    model = MyRNN(in_dim, hidden_dim, out_dim, num_layers=2,
                  bidirectional=True)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    model.train()
    batch_size = 2

    dataset = PyTorchDataset(X, Y, lengths)
    loader = data_utils.DataLoader(
        dataset, batch_size=batch_size, num_workers=1, shuffle=True)

    # Test if trining loop pass with no errors. The following code was adapted
    # from practical RNN training demo.
    for idx, (x, y, lengths) in enumerate(loader):
        # Sort by lengths indices
        sorted_lengths, indices = torch.sort(lengths.view(-1), dim=0,
                                             descending=True)
        sorted_lengths = sorted_lengths.long().numpy()
        # Get sorted batch
        x, y = x[indices], y[indices]
        # Trim outputs with max length
        y = y[:, :sorted_lengths[0]]

        x = Variable(x)
        y = Variable(y)
        h, c = model.init_hidden(len(sorted_lengths))
        optimizer.zero_grad()

        y_hat = model(x, sorted_lengths, h, c)
        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()
