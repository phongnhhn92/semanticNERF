import segmentation_models_pytorch as smp
import torch

model = smp.FPN('vgg19', in_channels=3)
mask = model.encoder(torch.ones([1, 3, 256, 256]))

print('a')