import torch.nn as nn
import torch


class AdaIN(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        eps = 1e-5
        mean_x = torch.mean(x, dim=[2, 3])
        mean_y = torch.mean(y, dim=[2, 3])

        std_x = torch.std(x, dim=[2, 3])
        std_y = torch.std(y, dim=[2, 3])

        mean_x = mean_x.unsqueeze(-1).unsqueeze(-1)
        mean_y = mean_y.unsqueeze(-1).unsqueeze(-1)

        std_x = std_x.unsqueeze(-1).unsqueeze(-1) + eps
        std_y = std_y.unsqueeze(-1).unsqueeze(-1) + eps

        out = (x - mean_x) / std_x * std_y + mean_y

        return out
