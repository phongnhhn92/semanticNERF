from torch import nn

class ColorLoss(nn.Module):
    def __init__(self,gamma = 0.1):
        super().__init__()
        self.loss = nn.MSELoss(reduction='mean')
        self.gamma = gamma

    def forward(self, inputs, targets):
        loss = self.gamma * self.loss(inputs['rgb_coarse'], targets)
        if 'rgb_fine' in inputs:
            loss += self.loss(inputs['rgb_fine'], targets)

        return loss
               

loss_dict = {'color': ColorLoss}