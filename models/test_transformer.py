import torch
import torch.autograd.profiler as profiler
from models.convTrans import CvT3D

torch.backends.cudnn.benchmark = True

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

v = CvT3D(
    num_layers = 6,
    emb_dim=[64, 128, 256, 128, 64, 32],
    emb_kernel=[3, 3, 3, 3, 3, 3],
    emb_stride=[2, 2, 2, 1, 1, 1,],
    proj_kernel=[3, 3, 3, 3, 3, 3],
    kv_proj_stride=[8, 4, 2, 2, 4 , 8],
    heads=[1, 2, 3, 3, 2, 1],
    depth=[1, 1, 1, 1, 1, 1],
    mlp_mult=[1, 1, 1, 1, 1, 1],
    dropout=0.0,
).to(device)
img = torch.randn(4, 3, 64, 32, 32).to(device)

pred = v(img)
print(pred.shape)



