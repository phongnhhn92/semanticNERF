"""
license: please refere to the SPADE repositry license
Copied from https://github.com/NVlabs/SPADE/blob/master/models/networks
"""
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .base_network import BaseNetwork
from .normalization import get_nonspade_norm_layer


class ConvEncoder(BaseNetwork):
    """ Same architecture as the image discriminator """

    def __init__(self, opts):
        super().__init__()
        self.opts = opts
        kw = 3
        pw = int(np.ceil((kw - 1.0) / 2))
        ndf = opts.ngf
        norm_layer = get_nonspade_norm_layer(opts, opts.norm_E)
        self.layer1 = norm_layer(nn.Conv2d(3, ndf, kw, stride=2, padding=pw))
        self.layer2 = norm_layer(
            nn.Conv2d(ndf * 1, ndf * 2, kw, stride=2, padding=pw))
        self.layer3 = norm_layer(
            nn.Conv2d(ndf * 2, ndf * 4, kw, stride=2, padding=pw))
        self.layer4 = norm_layer(
            nn.Conv2d(ndf * 4, ndf * 8, kw, stride=2, padding=pw))
        self.layer5 = norm_layer(
            nn.Conv2d(ndf * 8, ndf * 8, kw, stride=2, padding=pw))
        # if opts.crop_size >= 256:
        #     self.layer6 = norm_layer(
        #         nn.Conv2d(ndf * 8, ndf * 8, kw, stride=2, padding=pw))

        self.so = s0 = 8
        self.fc_mu = nn.Linear(ndf * 8 * s0 * s0, 256)
        self.fc_var = nn.Linear(ndf * 8 * s0 * s0, 256)

        self.actvn = nn.LeakyReLU(0.2, False)
        self.opts = opts
        # self.init_weights('xavier', gain=0.02)
        # self.init_weights('orthogonal')

    def forward(self, x):
        if x.size(2) != 256 or x.size(3) != 256:
            x = F.interpolate(x, size=(256, 256), mode='bilinear')
        feature_List = []
        x_ = self.layer1(x)
        feature_List.append(x_)
        x_ = self.layer2(self.actvn(x_))
        feature_List.append(x_)
        x_ = self.layer3(self.actvn(x_))
        feature_List.append(x_)
        x_ = self.layer4(self.actvn(x_))
        feature_List.append(x_)
        x_ = self.layer5(self.actvn(x_))
        feature_List.append(x_)
        # if self.opts.crop_size >= 256:
        #     x_ = self.layer6(self.actvn(x_))
        #     feature_List.append(x_)
        x_ = self.actvn(x_)

        x_ = x_.view(x_.size(0), -1)
        mu = self.fc_mu(x_)
        if self.opts.use_vae:
            logvar = self.fc_var(x_)
            z = self.reparameterize(mu, logvar)
            return z, mu, logvar, feature_List
        else:
            return mu,feature_List

    def reparameterize(self, mu, logvar):
        if not self.training:
            return mu
        else:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std) + mu
