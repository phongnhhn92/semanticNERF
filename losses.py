from torch import nn


class ColorLoss(nn.Module):
    def __init__(self, coef=1.0):
        super().__init__()
        self.coef = coef
        self.loss = nn.MSELoss(reduction='mean')
        self.segLoss = nn.CrossEntropyLoss()

    def forward(self, inputs, targets):
        loss = self.loss(inputs['rgb'], targets)
        return self.coef * loss
               

loss_dict = {'color': ColorLoss}