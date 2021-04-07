import torch
from torch import nn


class Embedding(nn.Module):
    def __init__(self, in_channels, N_freqs, logscale=True):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        in_channels: number of input channels (3 for both xyz and direction)
        """
        super(Embedding, self).__init__()
        self.N_freqs = N_freqs
        self.in_channels = in_channels
        self.funcs = [torch.sin, torch.cos]
        self.out_channels = in_channels * (len(self.funcs) * N_freqs + 1)

        if logscale:
            self.freq_bands = 2 ** torch.linspace(0, N_freqs - 1, N_freqs)
        else:
            self.freq_bands = torch.linspace(1, 2 ** (N_freqs - 1), N_freqs)

    def forward(self, x):
        """
        Embeds x to (x, sin(2^k x), cos(2^k x), ...) 
        Different from the paper, "x" is also in the output
        See https://github.com/bmild/nerf/issues/12

        Inputs:
            x: (B, self.in_channels)

        Outputs:
            out: (B, self.out_channels)
        """
        out = [x]
        for freq in self.freq_bands:
            for func in self.funcs:
                out += [func(freq * x)]

        return torch.cat(out, -1)


class IPEmbedding(nn.Module):
    def __init__(self, in_channels, N_freqs):
        """
        Intergrated positional encoding
        https://arxiv.org/abs/2103.13415
        """
        super(IPEmbedding, self).__init__()
        self.N_freqs = N_freqs
        self.in_channels = in_channels
        self.funcs = [torch.sin, torch.cos]
        self.out_channels = in_channels * (len(self.funcs) * N_freqs)
        self.freq_bands = 2 ** torch.linspace(0, N_freqs - 1, N_freqs)
        self.freq_bands2 = 4 ** torch.linspace(0, N_freqs - 1, N_freqs)

    def forward(self, mean, cov_diag):
        """
        Inputs:
            x: (B, self.in_channels)

        Outputs:
            out: (B, self.out_channels)
        """
        out = []
        for freq1, freq2 in zip(self.freq_bands, self.freq_bands2):
            for func in self.funcs:
                y = torch.cat([freq1 * mean])
                w = torch.cat([torch.exp(-0.5 * freq2 * cov_diag)])
                out += [func(y) * w]

        return torch.cat(out, -1)


class NeRF(nn.Module):
    def __init__(self,
                 D=8, W=256,
                 in_channels_xyzd=96,
                 skips=[4]):
        """
        D: number of layers for density (sigma) encoder
        W: number of hidden units in each layer
        in_channels_xyzd: number of input channels for xyz (2*16*3 = 96 by default)
        skips: add skip connection in the Dth layer
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.in_channels_xyzd = in_channels_xyzd
        self.skips = skips

        # xyz encoding layers
        for i in range(D):
            if i == 0:
                layer = nn.Linear(in_channels_xyzd, W)
            elif i in skips:
                layer = nn.Linear(W + in_channels_xyzd, W)
            else:
                layer = nn.Linear(W, W)
            layer = nn.Sequential(layer, nn.ReLU(True))
            setattr(self, f"xyz_encoding_{i + 1}", layer)

        # output layers
        self.sigma = nn.Sequential(nn.Linear(W, W // 2),
                                   nn.Linear(W // 2, 1))

        self.xyz_encoding_final = nn.Linear(W, W // 2)
        self.rgb = nn.Sequential(
            nn.Linear(W // 2, 3),
            nn.Sigmoid())

    def forward(self, x, sigma_only=False):
        """
        Encodes input (xyz+dir) to rgb+sigma (not ready to render yet).
        For rendering this ray, please see rendering.py

        Inputs:
            x: (B, self.in_channels_xyz(+self.in_channels_dir))
               the embedded vector of position and direction
            sigma_only: whether to infer sigma only. If True,
                        x is of shape (B, self.in_channels_xyz)

        Outputs:
            if sigma_ony:
                sigma: (B, 1) sigma
            else:
                out: (B, 4), rgb and sigma
        """

        xyz_ = x
        for i in range(self.D):
            if i in self.skips:
                xyz_ = torch.cat([x, xyz_], -1)
            xyz_ = getattr(self, f"xyz_encoding_{i + 1}")(xyz_)

        sigma = self.sigma(xyz_)
        if sigma_only:
            return sigma

        xyz_encoding_final = self.xyz_encoding_final(xyz_)
        rgb = self.rgb(xyz_encoding_final)

        out = torch.cat([rgb, sigma], -1)

        return out
