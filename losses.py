from torch import nn
import torch.nn.functional as F

class ColorLoss(nn.Module):
    def __init__(self, coef=1.0):
        super().__init__()
        self.coef = coef
        self.loss = nn.MSELoss(reduction='mean')
        self.segLoss = nn.CrossEntropyLoss()

    def forward(self, inputs, targetsRGB):
        loss = self.loss(inputs['rgb'], targetsRGB)
        return self.coef * loss
               
class SemanticLoss(nn.Module):
    def __init__(self, coef=1.0):
        super().__init__()
        self.coef = coef
        self.loss = nn.CrossEntropyLoss()
        self.num_classes = 13

    def forward(self, inputs, targetsSemantic):
        B,R,_ = inputs['semantic'].shape
        pred = inputs['semantic'].view(B*R,self.num_classes)
        target =targetsSemantic.view(B*R)
        loss =  F.cross_entropy(pred, target, ignore_index=self.num_classes)
        return self.coef * loss

loss_dict = {'color': ColorLoss, 'semantic': SemanticLoss}