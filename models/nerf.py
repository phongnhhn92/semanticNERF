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
        self.out_channels = in_channels*(len(self.funcs)*N_freqs+1)

        if logscale:
            self.freq_bands = 2**torch.linspace(0, N_freqs-1, N_freqs)
        else:
            self.freq_bands = torch.linspace(1, 2**(N_freqs-1), N_freqs)

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
                out += [func(freq*x)]

        return torch.cat(out, -1)


class NeRF(nn.Module):
    def __init__(self,
                 D=8, W=256,
                 in_channels_xyz=63, in_channels_dir=27,
                 in_channels_appearance=32,
                 in_channels_semantic=13,
                 skips=[4]):
        """
        D: number of layers for density (sigma) encoder
        W: number of hidden units in each layer
        in_channels_xyz: number of input channels for xyz (3+3*10*2=63 by default)
        in_channels_dir: number of input channels for direction (3+3*4*2=27 by default)
        skips: add skip connection in the Dth layer
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.in_channels_xyz = in_channels_xyz
        self.in_channels_dir = in_channels_dir
        self.in_channels_appearance = in_channels_appearance
        self.in_channels_semantic = in_channels_semantic
        self.skips = skips

        self.layer_appearance = nn.Linear(in_channels_appearance, W)
        self.layer_semantic = nn.Linear(in_channels_semantic, W)

        # xyz encoding layers
        for i in range(D):
            if i == 0:
                layer = nn.Linear(in_channels_xyz, W)
            elif i in skips:
                layer = nn.Linear(W+in_channels_xyz, W)
            else:
                layer = nn.Linear(W, W)
            layer = nn.Sequential(layer, nn.ReLU(True))
            setattr(self, f"xyz_encoding_{i+1}", layer)

        self.xyz_encoding_final = nn.Linear(W, W)

        # direction encoding layers
        self.style_encoding = nn.Sequential(
            nn.Linear(2 * W , W),
            nn.ReLU(True),
            nn.Linear(W, W // 2),
            nn.ReLU(True),
            nn.Linear(W // 2, in_channels_semantic),
            nn.LeakyReLU(True),
        )

        # direction encoding layers
        self.appearance_dir_encoding = nn.Sequential(
                                nn.Linear(2*W+in_channels_dir, W),
                                nn.ReLU(True),
                                nn.Linear(W, W // 2),
                                nn.ReLU(True),
                                nn.Linear(W // 2, 3),
                                nn.Sigmoid()
                            )

        # output layers
        self.sigma_color = nn.Linear(W, 1)

    def forward(self, x, style):
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

        input_xyz, input_dir = \
                torch.split(x, [self.in_channels_xyz, self.in_channels_dir], dim=-1)

        style_appearance, style_semantic = \
                torch.split(style,[self.in_channels_appearance, self.in_channels_semantic],dim=-1)

        xyz_ = input_xyz
        for i in range(self.D):
            if i == 0:
                xyz_ = torch.cat([xyz_],dim=-1)
            if i in self.skips:
                xyz_ = torch.cat([input_xyz, xyz_], -1)
            xyz_ = getattr(self, f"xyz_encoding_{i+1}")(xyz_)

        sigma_color = self.sigma_color(xyz_)
        xyz_encoding_final = self.xyz_encoding_final(xyz_)

        style_semantic = self.layer_semantic(style_semantic)
        semantic_encoding_input = torch.cat([xyz_encoding_final, style_semantic], -1)
        semantic = self.style_encoding(semantic_encoding_input)

        style_appearance = self.layer_appearance(style_appearance)
        appearance_dir_encoding_input = torch.cat([xyz_encoding_final, input_dir, style_appearance], -1)
        rgb = self.appearance_dir_encoding(appearance_dir_encoding_input)

        out = torch.cat([rgb, semantic, sigma_color], -1)

        return out