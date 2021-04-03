import torch

from models.convTrans import CvT3D

v = CvT3D(
    num_classes = 1000,
    num_layers = 3,
    emb_dim=[64, 128, 256],
    emb_kernel=[7, 3, 3],
    emb_stride=[4, 2, 2],
    proj_kernel=[3, 3, 3],
    kv_proj_stride=[2, 2, 2],
    heads=[1, 3, 4],
    depth=[1, 2, 10],
    mlp_mult=[4, 4, 4],
    dropout=0.
)

img = torch.randn(4, 3, 64, 32, 32)


pred = v(img) # (1, 1000)

print(pred.shape)

