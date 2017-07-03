import torch
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable


class CVAE(nn.Module):
    def __init__(self, n_in, n_latent, n_labels, use_label_in_encode=False):
        super(CVAE, self).__init__()
        if use_label_in_encode:
            n_encode_in = n_in + n_labels
        else:
            n_encode_in = n_in
        self.fc1 = nn.Linear(n_encode_in, 500)
        self.fc21 = nn.Linear(500, n_latent)
        self.fc22 = nn.Linear(500, n_latent)
        self.fc3 = nn.Linear(n_latent + n_labels, 500)
        self.fc4 = nn.Linear(500, n_in)
        self.relu = nn.ReLU()

        self.use_label_in_encode = use_label_in_encode
        self.n_in = n_in
        self.n_labels = n_labels

    def encode(self, x, y):
        if self.use_label_in_encode:
            data = torch.cat((x, y), -1)
        else:
            data = x
        h1 = self.relu(self.fc1(data))
        return self.fc21(h1), self.fc22(h1)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.cuda.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z, y):
        h3 = self.relu(self.fc3(torch.cat((z, y.view(-1, self.n_labels)), -1)))
        return self.fc4(h3)

    def forward(self, x, y):
        mu, logvar = self.encode(x.view(-1, self.n_in),
                                 y.view(-1, self.n_labels))
        z = self.reparametrize(mu, logvar)
        return self.decode(z, y), mu, logvar
