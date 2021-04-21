"""
license: please refere to the SPADE repositry license
Copied from https://github.com/NVlabs/SPADE/blob/master/models/networks
"""
import torch
import torch.nn as nn
from .base_network import BaseNetwork


class MLPEncoder(BaseNetwork):
    def __init__(self, opts):
        super().__init__()
        self.opts = opts
        ndf = opts.ngf

        self.so = s0 = 8
        self.fc_mu = nn.Linear(ndf * 8 * s0 * s0, 256)
        self.fc_var = nn.Linear(ndf * 8 * s0 * s0, 256)

        self.actvn = nn.LeakyReLU(0.2, False)
        self.opts = opts

    def forward(self, x):
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        if self.opts.use_vae:
            logvar = self.fc_var(x)
            z = self.reparameterize(mu, logvar)
            return z, mu, logvar
        else:
            return mu

    def reparameterize(self, mu, logvar):
        if not self.training:
            return mu
        else:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std) + mu
