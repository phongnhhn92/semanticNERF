from torch import nn
import torch

# KL Divergence loss used in VAE with an image encoder
class KLDLoss(nn.Module):
    def forward(self, mu, logvar, coef=0.001):
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return coef * kld_loss
class ColorLoss(nn.Module):
    def __init__(self, coef=1.0):
        super().__init__()
        self.coef = coef
        self.loss = nn.MSELoss(reduction='mean')
        self.segLoss = nn.CrossEntropyLoss()

    def forward(self, inputs, targets):
        loss = self.loss(inputs['rgb'], targets)
        return self.coef * loss
               

loss_dict = {'color': ColorLoss,'kl':KLDLoss}