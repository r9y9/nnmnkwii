from __future__ import division, print_function, absolute_import

import torch
import numpy as np
import math
from torch.autograd import Variable


def gaussian_nll(x, mean, logsigma2):
    sigma2 = torch.exp(logsigma2)  # sigma^2
    x_diff = x - mean
    x_power = x_diff * x_diff / sigma2 * -0.5
    return (logsigma2 + math.log(2 * math.pi)) * 0.5 - x_power


def vae_loss_function(recon_x, x, mu, logvar):
    # Hyper parameter for observation model
    sigma2 = 1.0
    logsigma2 = np.log(sigma2)

    v = Variable(torch.ones(x.size())) * logsigma2
    v = v.cuda()
    GLL = torch.sum(gaussian_nll(x, recon_x, v))

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    return GLL + KLD
