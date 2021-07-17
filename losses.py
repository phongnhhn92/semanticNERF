from torch import nn
import torch
from torch.nn import functional as F

class ContentLoss(nn.Module):
    
    def __init__(self,coef = 1.0):
        super(ContentLoss, self).__init__()
        self.loss = nn.MSELoss(reduction='mean')
        self.coef = coef
    def forward(self, input, target):
        target = target.detach()
        loss = self.loss(input, target)
        return self.coef * loss

def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)
class StyleLoss(nn.Module):
    
    def __init__(self,coef = 1000000):
        super(StyleLoss, self).__init__()
        self.loss = nn.MSELoss(reduction='mean')
        self.coef = coef

    def forward(self, input,target):
        input = gram_matrix(input)
        target = gram_matrix(target.detach())
        loss = self.loss(input,target)
        return self.coef * loss


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