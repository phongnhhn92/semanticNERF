from torch import nn

class ColorLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss(reduction='mean')

    def forward(self, inputs, targets,gamma = 0.1):
        loss = gamma * self.loss(inputs['rgb_coarse'], targets)
        if 'rgb_fine' in inputs:
            loss += self.loss(inputs['rgb_fine'], targets)

        return self.coef * loss
               

loss_dict = {'color': ColorLoss}