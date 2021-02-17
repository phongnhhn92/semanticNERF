from torch import nn
import robust_loss_pytorch.general
import numpy as np

class ColorLoss(nn.Module):
    def __init__(self, coef=1):
        super().__init__()
        self.coef = coef
        #self.loss = nn.MSELoss(reduction='mean')
        self.loss = nn.SmoothL1Loss(reduction='mean')

        # Add params to train
        #self.adaptive = robust_loss_pytorch.adaptive.AdaptiveLossFunction(num_dims = 1, float_dtype=np.float32, device='cpu')


    def forward(self, inputs, targets):
        loss = self.loss(inputs['rgb_coarse'], targets)
        if 'rgb_fine' in inputs:
            loss += self.loss(inputs['rgb_fine'], targets)

        return self.coef * loss
               

loss_dict = {'color': ColorLoss}