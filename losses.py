from torch import nn


class ColorLoss(nn.Module):
    def __init__(self, coef=1):
        super().__init__()
        self.coef = coef
        self.loss = nn.MSELoss(reduction='mean')
        self.segLoss = nn.CrossEntropyLoss()

    def forward(self, inputs, targets,segs):
        loss = self.loss(inputs['rgb_coarse'], targets)
        loss += 0.1 * self.segLoss(inputs['feature_coarse'], segs)
        if 'rgb_fine' in inputs:
            loss += self.loss(inputs['rgb_fine'], targets)
            loss += 0.1 * self.segLoss(inputs['feature_fine'], segs)

        return self.coef * loss
               

loss_dict = {'color': ColorLoss}