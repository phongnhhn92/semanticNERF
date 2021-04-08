import torch
from torch import nn, einsum
from einops import rearrange

torch.backends.cudnn.benchmark = True

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

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
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Conv3d(dim * mult, dim, 1),
            nn.ReLU(True),
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
            nn.ReLU(True),
            nn.Conv3d(dim_in, dim_out, kernel_size = 1, bias = bias),
            nn.ReLU(True),

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
            nn.ReLU(True),
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
            num_layers=6,
            input_dim=63,
            dir_dim=27,
            emb_dim=[64, 128, 256, 128, 64, 32],
            emb_kernel=[3, 3, 3, 3, 3, 3],
            emb_stride=[2, 2, 2, 1, 1, 1],
            proj_kernel=[3, 3, 3, 3, 3, 3],
            kv_proj_stride=[8, 4, 2, 2, 4, 8],
            heads=[1, 2, 3, 3, 2, 1],
            depth=[1, 1, 1, 1, 1, 1],
            mlp_mult=[1, 1, 1, 1, 1, 1],
            dropout=0.0,
    ):
        super().__init__()
        kwargs = dict(locals())
        self.num_layers = num_layers
        dim = input_dim
        for i in range(num_layers):

            conv_layer = nn.Sequential(nn.Conv3d(dim, emb_dim[i], kernel_size=emb_kernel[i], padding=(emb_kernel[i] // 2),
                          stride=emb_stride[i]),nn.ReLU(True))

            transformer = Transformer(dim=emb_dim[i], proj_kernel=proj_kernel[i],
                            kv_proj_stride=kv_proj_stride[i], depth=depth[i], heads=heads[i],
                            mlp_mult=mlp_mult[i], dropout=dropout)
            if i < 3:
                setattr(self, f"layer_{i+1}", nn.Sequential(conv_layer,transformer))
            else:
                upsample = nn.Upsample(scale_factor=2,align_corners=True,mode='trilinear')
                setattr(self, f"layer_{i + 1}", nn.Sequential(upsample,conv_layer, transformer))

            dim = emb_dim[i]

        self.sigma_layer = nn.Conv3d(dim, 1, kernel_size=3, padding=1,stride=1)
        self.rgb_layer = nn.Sequential(nn.Conv3d(dim + dir_dim, dim //2 , kernel_size=3, padding=1,stride=1),
                                       nn.ReLU(True),
                                       nn.Conv3d(dim //2 , 3, kernel_size=3, padding=1,stride=1),
                                       nn.Sigmoid())


    def forward(self, xyz, dir):
        layer1 = self.layer_1(xyz)
        layer2 = self.layer_2(layer1)
        out = self.layer_3(layer2)
        out = self.layer_4(out)
        out = self.layer_5(out + layer2)
        del layer2
        out = self.layer_6(out + layer1)
        del layer1

        sigma = self.sigma_layer(out)
        rgb = self.rgb_layer(torch.cat((out,dir),dim =1))

        return torch.cat((rgb,sigma),dim = 1)

v = CvT3D().to(device)
img = torch.randn(4, 3, 64, 32, 32).to(device)

pred = v(img)
print(pred.shape)



