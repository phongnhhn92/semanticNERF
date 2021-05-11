from torch import nn
import torch

class ColorLoss(nn.Module):
    def __init__(self, coef=1.0):
        super().__init__()
        self.coef = coef
        self.loss = nn.MSELoss(reduction='mean')
        self.segLoss = nn.CrossEntropyLoss()

    def forward(self, inputs, targets):
        loss = self.loss(inputs['rgb'], targets)
        return self.coef * loss
               
# KL Divergence loss used in VAE with an image encoder
class KLDLoss(nn.Module):
    def forward(self, mu, logvar):
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return kld_loss

loss_dict = {'color': ColorLoss, 'kl': KLDLoss}