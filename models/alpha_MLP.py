from torch import nn

class Alpha_MLP(nn.Module):
    def __init__(self,
                 W=1024,
                 in_channels = 32,
                 out_channels = 32 * 128):
        """
        D: number of layers for density (sigma) encoder
        W: number of hidden units in each layer
        in_channels_xyz: number of input channels for xyz (3+3*10*2=63 by default)
        in_channels_dir: number of input channels for direction (3+3*4*2=27 by default)
        skips: add skip connection in the Dth layer
        """
        super(Alpha_MLP, self).__init__()
        self.W = W
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.alpha = nn.Sequential(
                        nn.Linear(in_channels, W),
                        nn.ReLU(True),
                        nn.Linear(W, out_channels),
                        nn.LeakyReLU(True))

    def forward(self, x):
        out = self.alpha(x)
        return out