"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_network import BaseNetwork
from .normalization import get_nonspade_norm_layer
from .architecture import ResnetBlock as ResnetBlock
from .architecture import SPADEResnetBlock as SPADEResnetBlock
from einops import rearrange
from torch import nn, einsum

class DepthWiseConv(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, padding, stride, bias = True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size = kernel_size, padding = padding, groups = dim_in, stride = stride, bias = bias),
            nn.BatchNorm2d(dim_in),
            nn.ReLU(True),
            nn.Conv2d(dim_in, dim_out, kernel_size = 1, bias = bias),
            nn.ReLU(True),

        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, proj_kernel, kv_proj_stride = 1, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        padding = proj_kernel // 2
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)

        self.to_q = DepthWiseConv(dim, inner_dim, 3, padding = padding, stride = 1, bias = False)
        self.to_kv = DepthWiseConv(dim, inner_dim * 2, 3, padding = padding, stride = kv_proj_stride, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim, 1),
            nn.ReLU(True),
            nn.Dropout(dropout)
        )

    def forward(self, x, features):
        b,c,_,y = x.shape
        h = self.heads
        q = self.to_q(x)
        k,v = self.to_kv(features).chunk(2,dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h d) x y -> (b h) (x y) d', h = h), (q, k, v))

        dots = einsum('b i d, b j d -> b i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) (x y) d -> b (h d) x y', h = h, y = y)
        return self.to_out(out)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, features):
        x = rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        x = rearrange(x, 'b h w c -> b c h w')
        return self.fn(x, features)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1),
            nn.ReLU(True),
            nn.Dropout(dropout),
        )
    def forward(self, x, features=None):
        return self.net(x)

class Transformer(nn.Module):
    def __init__(self, dim, proj_kernel, kv_proj_stride, heads, depth=1, dim_head = 64, mlp_mult = 1, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, proj_kernel = proj_kernel, kv_proj_stride = kv_proj_stride, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_mult, dropout = dropout))
            ]))
    def forward(self, x, feature):
        for attn, ff in self.layers:
            x = attn(x, feature) + x
            x = ff(x, feature) + x
        return x

class SPADEGenerator(BaseNetwork):
    def __init__(self, opts, no_tanh=True):
        '''
            Since we extract layered features
        '''
        super().__init__()
        # opts = self.modify_commandline_options(opts, True)
        self.opts = opts
        self.no_tanh = no_tanh
        num_context_chans = 0
        nf = opts.ngf
        self.sw, self.sh = self.compute_latent_vector_size(opts)
        if opts.use_vae:
            self.fc = nn.Linear(opts.z_dim, 8 * nf * self.sw * self.sh)
        else:
            if opts.use_instance_mask:
                self.fc = nn.Conv2d(self.opts.embedding_size+1, 16 * nf, 3, padding=1)
            else:
                self.fc = nn.Conv2d(self.opts.embedding_size, 16 * nf, 3, padding=1)
        self.head_0 = SPADEResnetBlock((8 * nf), 8 * nf, opts)
        self.attention_0 = Transformer(dim=8*nf,proj_kernel=3,kv_proj_stride=1,heads=8)

        self.G_middle_0 = SPADEResnetBlock((8 * nf), 8 * nf, opts)
        self.G_middle_1 = SPADEResnetBlock((8 * nf), 8 * nf, opts)
        self.attention_1 = Transformer(dim=8 * nf, proj_kernel=3, kv_proj_stride=1, heads=8)

        self.up_0 = SPADEResnetBlock((8 * nf), 4 * nf, opts)
        self.attention_2 = Transformer(dim=4 * nf, proj_kernel=3, kv_proj_stride=2, heads=6)

        self.up_1 = SPADEResnetBlock((4 * nf), 2 * nf, opts)
        self.attention_3 = Transformer(dim=2 * nf, proj_kernel=3, kv_proj_stride=2, heads=4)

        self.up_2 = SPADEResnetBlock((2 * nf), 1 * nf, opts)
        self.attention_4 = Transformer(dim=1 * nf, proj_kernel=3, kv_proj_stride=4, heads=2)

        self.up_3 = SPADEResnetBlock((1 * nf), 1 * nf, opts)
        final_nc = nf

        if opts.num_upsampling_layers == 'most':
            self.up_4 = SPADEResnetBlock((1 * nf), nf // 2, opts)
            final_nc = nf // 2

        self.conv_img = nn.Conv2d(final_nc, opts.num_out_channels, 3, padding=1)

        self.up = nn.Upsample(scale_factor=2)
        # self.init_weights('xavier', gain=0.02)

    def compute_latent_vector_size(self, opts):
        if opts.num_upsampling_layers == 'normal':
            num_up_layers = 5
        elif opts.num_upsampling_layers == 'more':
            num_up_layers = 6
        elif opts.num_upsampling_layers == 'most':
            num_up_layers = 7
        else:
            raise ValueError('opts.num_upsampling_layers [%s] not recognized' %
                             opts.num_upsampling_layers)

        sw = opts.img_wh[0] // (2**num_up_layers)
        sh = opts.img_wh[1] // (2**num_up_layers)

        return sw, sh

    def forward(self, seg_map, features, z=None):
        seg = seg_map
        if self.opts.use_vae:
            x = self.fc(z)
        else:
            x = F.interpolate(seg, size=(self.sh, self.sw))
            x = self.fc(x)
        x = x.view(-1, 8 * self.opts.ngf, self.sh, self.sw)
        x = self.head_0(x, seg)
        x = self.attention_0(x,features[4])
        x = self.up(x)

        x = self.G_middle_0(x, seg)
        x = self.G_middle_1(x, seg)
        x = self.attention_1(x, features[3])
        x = self.up(x)

        x = self.up_0(x, seg)
        x = self.attention_2(x, features[2])
        x = self.up(x)

        x = self.up_1(x, seg)
        x = self.attention_3(x, features[1])
        x = self.up(x)

        x = self.up_2(x, seg)
        x = self.attention_4(x, features[0])
        x = self.up(x)

        x = self.up_3(x, seg)
        x = self.conv_img(F.leaky_relu(x, 2e-1))
        if self.no_tanh:
            return x
        else:
            x = torch.tanh(x)
            return x
