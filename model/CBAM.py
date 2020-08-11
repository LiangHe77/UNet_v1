import torch
import torch.nn as nn

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, kernels_per_layer=1):
        super(DepthwiseSeparableConv, self).__init__()

        self.depthwise = nn.Conv2d(in_channels, in_channels*kernels_per_layer,
                                   kernel_size=kernel_size,
                                   padding=padding,
                                   groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels*kernels_per_layer, out_channels,
                                   kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class DoubleDSConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleDSConv, self).__init__()

        self.double_ds_conv = nn.Sequential(
            DepthwiseSeparableConv(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_ds_conv(x)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()

        self.in_channels = in_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.MLP = nn.Sequential(
            Flatten(),
            nn.Linear(in_channels, in_channels//reduction_ratio),
            nn.ReLU(),
            nn.Linear(in_channels//reduction_ratio, in_channels)
        )

    def forward(self, x):
        avg_values = self.avg_pool(x)
        max_values = self.max_pool(x)
        out = self.MLP(avg_values) + self.MLP(max_values)
        scale = x * torch.sigmoid(out).unsqueeze(2).unsqueeze(3).expand_as(x)
        return scale

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'

        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(1)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out,_ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        out = self.bn(out)
        scale = x * torch.sigmoid(out)
        return scale

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention(in_channels, reduction_ratio=reduction_ratio)
        self.spatial_att = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        out = self.channel_att(x)
        out = self.spatial_att(out)
        return out