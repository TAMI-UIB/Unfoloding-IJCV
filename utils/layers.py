import torch
from einops import rearrange

from torch import nn
import torch.nn.functional as F

class ConvRelu(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size):
        super(ConvRelu, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2, bias=False), nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x
class ResBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResBlock, self).__init__()
        features = in_channels
        kernel_size=3
        self.conv1 = nn.Conv2d(in_channels, features, kernel_size, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(features, in_channels, kernel_size, padding=1)
    def forward(self, x):
        features = self.conv1(x)
        features = self.relu(features)
        features = self.conv2(features)
        return self.relu(features + x)

class HeadAttention(torch.nn.Module):
    def __init__(self, emb_dim, bias=False):
        super(HeadAttention, self).__init__()
        self.emb_dim = emb_dim
        self.norm= nn.InstanceNorm2d(num_features=emb_dim)
        self.phi_queries = nn.Linear(emb_dim, 48,  bias=bias)
        self.theta_keys = nn.Linear(emb_dim, 48,  bias=bias)
        self.g_values = nn.Linear(emb_dim, emb_dim, bias=bias)

    def forward(self, keys, queries, values):
        _, _, h, w = queries.size()
        keys = self.norm(keys)
        queries = self.norm(queries)
        values = values

        queries = rearrange(queries, 'b c h w -> b (h w) c')
        keys = rearrange(keys, 'b c h w -> b (h w) c')
        values = rearrange(values, 'b c h w -> b (h w) c')

        phi = self.phi_queries(queries)
        theta = self.theta_keys(keys)
        g = self.g_values(values)

        head = F.scaled_dot_product_attention(query=phi, key=theta, value=g, attn_mask=None, dropout_p=0)
        head = rearrange(head, 'b (h w) c -> b c h w', h=h, w=w)
        return head


class MultiHeadAttention(nn.Module):
    def __init__(self, hs_channels, patch_size=8, features=64):
        super(MultiHeadAttention, self).__init__()


        self.patch_size = patch_size
        emb_dim = hs_channels * patch_size * patch_size
        self.head_attention1 = HeadAttention(emb_dim)
        self.head_attention2 = HeadAttention(emb_dim)
        self.head_attention3 = HeadAttention(emb_dim)
        self.mlp = nn.Sequential(nn.Conv2d(hs_channels*3, features, kernel_size=1, bias=False),
                                 nn.GELU(),
                                 nn.Conv2d(features, hs_channels, kernel_size=1, bias=False),
                                 nn.GELU(),
                                 )

        self.up = nn.PixelShuffle(upscale_factor=patch_size)

    def forward(self, u, pan):
        b, c, h, w = u.size()
        h1 = h // self.patch_size
        w1 = w // self.patch_size
        u_emb = F.unfold(u, kernel_size=self.patch_size, stride=self.patch_size)
        pan_emb = F.unfold(pan, kernel_size=self.patch_size, stride=self.patch_size)
        u_emb = rearrange(u_emb, "b l (h1 w1) -> b l h1 w1", w1=w1, h1=h1)
        pan_emb = rearrange(pan_emb, "b l (h1 w1) -> b l h1 w1", w1=w1, h1=h1)
        head1 = self.up(self.head_attention1(keys=u_emb, queries=u_emb, values=u_emb))
        head2 = self.up(self.head_attention2(keys=u_emb, queries=pan_emb, values=u_emb))
        head3 = self.up(self.head_attention3(keys=pan_emb, queries=pan_emb, values=u_emb))
        mha = torch.cat([head1, head2, head3], dim=1)
        return self.mlp(mha)

class NLBPUNetFormer(torch.nn.Module):
    def __init__(self, hs_channels, features, patch_size, kernel_size=3):
        super(NLBPUNetFormer, self).__init__()
        self.patch_size = patch_size
        self.features = features

        self.Nonlocal = MultiHeadAttention(hs_channels=hs_channels, patch_size=patch_size, features=features)

        self.feat_u = nn.Conv2d(in_channels=hs_channels, out_channels=features,
                                    kernel_size=kernel_size, stride=1, bias=False, padding=kernel_size // 2)
        self.feat_u_nl = nn.Conv2d(in_channels=hs_channels, out_channels=features,
                                kernel_size=kernel_size, stride=1, bias=False, padding=kernel_size // 2)

        self.feat_pan = nn.Conv2d(in_channels=hs_channels, out_channels=features, kernel_size=kernel_size, stride=1,
                                        bias=False, padding=kernel_size // 2)
        self.recon = ConvRelu(in_channels=features, out_channels=hs_channels,
                                   kernel_size=kernel_size)
        self.feat_aux= nn.Conv2d(features+features+features, features, kernel_size=3, padding=1)
        self.residual = nn.Sequential(*[ResBlock(in_channels=features) for _ in range(3)])

    def forward(self, u, pan):
        # Multi Attention Component
        u_nl = self.Nonlocal(u, pan)
        u_nl = torch.roll(u_nl, shifts=(self.patch_size//2, self.patch_size//2), dims=(2, 3))
        pan = torch.roll(pan,
                         shifts=(self.patch_size // 2, self.patch_size // 2),
                         dims=(2,3))
        u_nl = self.Nonlocal(u_nl, pan)
        u_nl = torch.roll(u_nl, shifts=(-self.patch_size // 2, -self.patch_size // 2), dims=(2, 3))
        pan = torch.roll(pan, shifts=(
            -self.patch_size // 2, -self.patch_size // 2), dims=(2,3))
        # Residual Component
        u_feat = self.feat_u(u)
        u_feat_nl = self.feat_u_nl(u_nl)
        pan_features = self.feat_pan(pan)
        u_aux = torch.cat([u_feat_nl, u_feat, pan_features], dim=1)
        u_aux= self.feat_aux(u_aux)
        res = self.residual(u_aux)
        return self.recon(res)

class ResNetU(nn.Module):
    def __init__(self, channels, features, index, stages, num_blocks, norm_layer = None):
        super(ResNetU, self).__init__()
        self.get_features = nn.Conv2d(in_channels=channels,out_channels=features,kernel_size=3,padding=1)
        self.resnet = nn.Sequential(*[ResBlock(features) for _ in range(num_blocks)])
        self.get_channel = nn.Conv2d(in_channels=features,out_channels=channels,kernel_size=3,padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigm = torch.nn.Sigmoid()

    def forward(self, x):
        res = self.get_features(x)
        res = self.resnet(res)
        res = self.get_channel(res)

        return self.sigm(res+x)