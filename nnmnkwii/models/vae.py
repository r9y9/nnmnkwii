import torch
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable


class VAE(nn.Module):
    def __init__(self, n_in, n_latent):
        super(VAE, self).__init__()

        self.n_in = n_in
        self.n_latent = n_latent

        self.fc1 = nn.Linear(n_in, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc31 = nn.Linear(512, n_latent)
        self.fc32 = nn.Linear(512, n_latent)
        self.fc4 = nn.Linear(n_latent, 512)
        self.fc5 = nn.Linear(512, 512)
        self.fc6 = nn.Linear(512, n_in)

        self.relu = nn.ReLU()

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        h2 = self.relu(self.fc2(h1))
        return self.fc31(h2), self.fc32(h2)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.cuda.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h4 = self.relu(self.fc4(z))
        h5 = self.relu(self.fc5(h4))
        return self.fc6(h5)

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.n_in))
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar
