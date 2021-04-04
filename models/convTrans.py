import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helper methods

def group_dict_by_key(cond, d):
    return_val = [dict(), dict()]
    for key in d.keys():
        match = bool(cond(key))
        ind = int(not match)
        return_val[ind][key] = d[key]
    return (*return_val,)

def group_by_key_prefix_and_remove_prefix(prefix, d):
    kwargs_with_prefix, kwargs = group_dict_by_key(lambda x: x.startswith(prefix), d)
    kwargs_without_prefix = dict(map(lambda x: (x[0][len(prefix):], x[1]), tuple(kwargs_with_prefix.items())))
    return kwargs_without_prefix, kwargs

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        x = rearrange(x, 'b c z h w -> b z h w c')
        x = self.norm(x)
        x = rearrange(x, 'b z h w c -> b c z h w')
        return self.fn(x, **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(dim, dim * mult, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv3d(dim * mult, dim, 1),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class DepthWiseConv3d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, padding, stride, bias = True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(dim_in, dim_in, kernel_size = kernel_size, padding = padding, groups = dim_in, stride = stride, bias = bias),
            nn.BatchNorm3d(dim_in),
            nn.Conv3d(dim_in, dim_out, kernel_size = 1, bias = bias)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, proj_kernel, kv_proj_stride, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        padding = proj_kernel // 2
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)

        self.to_q = DepthWiseConv3d(dim, inner_dim, 3, padding = padding, stride = 1, bias = False)
        self.to_kv = DepthWiseConv3d(dim, inner_dim * 2, 3, padding = padding, stride = kv_proj_stride, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv3d(inner_dim, dim, 1),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        #shape = x.shape
        #b, n, d, _, y, h = *shape, self.heads
        b,c,z,_,y = x.shape
        h = self.heads
        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim = 1))
        q, k, v = map(lambda t: rearrange(t, 'b (h d) z x y -> (b h) (z x y) d', h = h), (q, k, v))

        dots = einsum('b i d, b j d -> b i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) (z x y) d -> b (h d) z x y', h = h, y = y,z = z)
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, proj_kernel, kv_proj_stride, depth, heads, dim_head = 64, mlp_mult = 4, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, proj_kernel = proj_kernel, kv_proj_stride = kv_proj_stride, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_mult, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class CvT3D(nn.Module):
    def __init__(
        self,
        *,
        num_classes,
        num_layers,
        emb_dim= [64,128,256,256,128,64],
        emb_kernel=[7,3,3,3,3,3],
        emb_stride=[4,2,2,1,1,1],
        proj_kernel=[3,3,3,3,3,3],
        kv_proj_stride=[2,2,2,1,1,1],
        heads=[1,3,4,4,3,1],
        depth=[1,2,3,3,2,1],
        mlp_mult=[4,4,4,4,4,4],
        dropout = 0.
    ):
        super().__init__()
        kwargs = dict(locals())
        self.num_layers = num_layers
        dim = 3

        for i in range(num_layers):

            conv_layer = nn.Conv3d(dim, emb_dim[i], kernel_size=emb_kernel[i], padding=(emb_kernel[i] // 2),
                          stride=emb_stride[i])

            transformer = Transformer(dim=emb_dim[i], proj_kernel=proj_kernel[i],
                            kv_proj_stride=kv_proj_stride[i], depth=depth[i], heads=heads[i],
                            mlp_mult=mlp_mult[i], dropout=dropout)
            if i < 3:
                setattr(self, f"layer_{i+1}", nn.Sequential(conv_layer,transformer))
            else:
                upsample = nn.Upsample(scale_factor=2,align_corners=True,mode='trilinear')
                setattr(self, f"layer_{i + 1}", nn.Sequential(upsample,conv_layer, transformer))

            dim = emb_dim[i]


    def forward(self, x):
        layer1 = self.layer_1(x)
        layer2 = self.layer_2(layer1)
        layer3 = self.layer_3(layer2)
        layer4 = self.layer_4(layer3)
        layer5 = self.layer_5(layer4 + layer2)
        layer6 = self.layer_6(layer5 + layer1)
        return layer6
