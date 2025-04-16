import torch.autograd.profiler as profiler
import spconv.pytorch as spconv
from einops import rearrange
import torch.nn as nn
from torch import einsum
import torch.nn.functional as F
import torch
from torchvision.models import resnet18

def single_conv(in_channels, out_channels, indice_key=None):
    return spconv.SparseSequential(
        spconv.SubMConv3d(in_channels,
                          out_channels,
                          1,
                          bias=False,
                          indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
    )


def double_conv(in_channels, out_channels, indice_key=None):
    # tmp = spconv.SubMConv3d(in_channels,
    #                       out_channels,
    #                       3,
    #                       bias=False,
    #                       indice_key=indice_key)
    return spconv.SparseSequential(
        spconv.SubMConv3d(in_channels,
                          out_channels,
                          3,
                          bias=False,
                          indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
        spconv.SubMConv3d(out_channels,
                          out_channels,
                          3,
                          bias=False,
                          indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
    )


def triple_conv(in_channels, out_channels, indice_key=None):
    return spconv.SparseSequential(
        spconv.SubMConv3d(in_channels,
                          out_channels,
                          3,
                          bias=False,
                          indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
        spconv.SubMConv3d(out_channels,
                          out_channels,
                          3,
                          bias=False,
                          indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
        spconv.SubMConv3d(out_channels,
                          out_channels,
                          3,
                          bias=False,
                          indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
    )


def stride_conv(in_channels, out_channels, indice_key=None):
    return spconv.SparseSequential(
        spconv.SparseConv3d(in_channels,
                            out_channels,
                            3,
                            2,
                            padding=1,
                            bias=False,
                            indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01), nn.ReLU())
    
class SparseConvNet(nn.Module):
    """Find the corresponding 3D feature of query point along the ray
    
    Attributes:
        conv: sparse convolutional layer 
        down: sparse convolutional layer with downsample 
    """
    def __init__(self, num_layers=2):
        super(SparseConvNet, self).__init__()
        self.num_layers = num_layers

        # self.conv0 = double_conv(3, 16, 'subm0')
        # self.down0 = stride_conv(16, 32, 'down0')

        # self.conv1 = double_conv(32, 32, 'subm1')
        # self.down1 = stride_conv(32, 64, 'down1')

        # self.conv2 = triple_conv(64, 64, 'subm2')
        # self.down2 = stride_conv(64, 128, 'down2')

        self.conv0 = double_conv(32, 32, 'subm0')
        self.down0 = stride_conv(32, 32, 'down0')

        self.conv1 = double_conv(32, 32, 'subm1')
        self.down1 = stride_conv(32, 64, 'down1')

        self.conv2 = triple_conv(64, 64, 'subm2')
        self.down2 = stride_conv(64, 96, 'down2')

        self.conv3 = triple_conv(96, 96, 'subm3')
        self.down3 = stride_conv(96, 96, 'down3')

        self.conv4 = triple_conv(96, 96, 'subm4')

        self.channel = 32

    def forward(self, x, point_normalied_coords):
        """Find the corresponding 3D feature of query point along the ray.

        Args:
            x: Sparse Conv Tensor
            point_normalied_coords: Voxel grid coordinate, integer normalied to [-1, 1]
        
        Returns:
            features: Corresponding 3D feature of query point along the ray
        """
        features = []

        net = self.conv0(x)
        net = self.down0(net)

        # point_normalied_coords = point_normalied_coords.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        if self.num_layers > 1:
            net = self.conv1(net)
            net1 = net.dense()
            # torch.Size([1, 32, 1, 1, 4096])
            feature_1 = F.grid_sample(net1, point_normalied_coords, padding_mode='zeros', align_corners=True)
            features.append(feature_1)
            self.channel = 32
            net = self.down1(net)
        
        if self.num_layers > 2:
            net = self.conv2(net)
            net2 = net.dense()
            # torch.Size([1, 64, 1, 1, 4096])
            feature_2 = F.grid_sample(net2, point_normalied_coords, padding_mode='zeros', align_corners=True)
            features.append(feature_2)
            self.channel = 64
            net = self.down2(net)
        
        if self.num_layers > 3:
            net = self.conv3(net)
            net3 = net.dense()
            # 128
            feature_3 = F.grid_sample(net3, point_normalied_coords, padding_mode='zeros', align_corners=True)
            features.append(feature_3)
            self.channel = 128
            net = self.down3(net)
        
        if self.num_layers > 4:
            net = self.conv4(net)
            net4 = net.dense()
            # 256
            feature_4 = F.grid_sample(net4, point_normalied_coords, padding_mode='zeros', align_corners=True)
            features.append(feature_4)

        features = torch.cat(features, dim=1)
        features = features.view(features.size(0), -1, features.size(4)).transpose(1,2)

        return features
    
    
#----------------------------------------------------------------------------

class PositionalEncoding(torch.nn.Module):
    """
    Implement NeRF's positional encoding
    """

    def __init__(self, num_freqs=6, d_in=3, freq_factor=None, include_input=True):
        super().__init__()
        self.num_freqs = num_freqs
        self.d_in = d_in
        # self.freqs = freq_factor * 2.0 ** torch.arange(0, num_freqs)
        self.freqs = 2.**torch.linspace(0., num_freqs-1, steps=num_freqs)
        self.d_out = self.num_freqs * 2 * d_in
        self.include_input = include_input
        if include_input:
            self.d_out += d_in
        # f1 f1 f2 f2 ... to multiply x by
        self.register_buffer(
            "_freqs", torch.repeat_interleave(self.freqs, 2).view(1, -1, 1)
        )
        # 0 pi/2 0 pi/2 ... so that
        # (sin(x + _phases[0]), sin(x + _phases[1]) ...) = (sin(x), cos(x)...)
        _phases = torch.zeros(2 * self.num_freqs)
        _phases[1::2] = torch.pi * 0.5
        self.register_buffer("_phases", _phases.view(1, -1, 1))

    def forward(self, x):
        """
        Apply positional encoding (new implementation)
        :param x (batch, self.d_in)
        :return (batch, self.d_out)
        """
        with profiler.record_function("positional_enc"):
            embed = x.unsqueeze(1).repeat(1, self.num_freqs * 2, 1)
            embed = torch.sin(torch.addcmul(self._phases, embed, self._freqs))
            if x.shape[0]==0:
                embed = embed.view(x.shape[0], self.num_freqs*6)
            else:
                embed = embed.view(x.shape[0], -1)
                
            if self.include_input:
                embed = torch.cat((x, embed), dim=-1)
            return embed

#----------------------------------------------------------------------------

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1) # torch.Size([30786, 3, 768])
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim=32, depth=1, heads=3, dim_head=16, mlp_dim=32, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)
        return x

#----------------------------------------------------------------------------


class ResNet18Classifier(nn.Module):
    def __init__(self, *args, **kwargs):
        super(ResNet18Classifier, self).__init__()
        self.backbone = resnet18(pretrained=True)

    def forward(self, x, extract_feature=False):
        # x = self.backbone(x)
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        if not extract_feature:
            x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        if extract_feature:
            return x
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)

        return x

#----------------------------------------------------------------------------