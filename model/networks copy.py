import torch
import torch.nn as nn
from torchvision import models
import math
from einops.einops import rearrange
import functools

VGG19_FEATURES = models.vgg19(pretrained=True).features
CONV3_3_IN_VGG_19 = VGG19_FEATURES[0:15].cuda()

def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.normal_(m.weight, 0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    if isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1, 0.02)
        nn.init.zeros_(m.bias)
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = nn.BatchNorm2d
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

class SEAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
    
class DenseBlock(nn.Module):
    """
    实现DenseNet中的密集连接结构
    输入: (N, C_in, H, W)
    输出: (N, C_in+C_out, H, W)
    """
    def __init__(self, in_channels, out_channels):
        super(DenseBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
            
        )
        self.se_attention = SEAttention(out_channels)
        self.convhalf = nn.Sequential(
            nn.Conv2d(out_channels + in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x_1 = torch.cat([x, self.conv(x)], 1)
        x = self.convhalf(x_1)
        out = self.se_attention(x)
        return x
        # return self.conv(x)

class Encoder(nn.Module):
    """
    由DenseBlock和下采样层组成,用于提取特征
    输入: (N, C_in, H, W)
    输出: [(N, C_out, H/2, W/2), (N, C_out, H, W)]
    """
    def __init__(self, in_channels, out_channels):
        super(Encoder, self).__init__()
        self.dense = DenseBlock(in_channels, out_channels)
        self.down = nn.Conv2d(out_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        x = self.dense(x)
        print("x:", x.shape)
        
        down = self.down(x)
        print("down:", down.shape)
        return down, x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += residual
        out = self.relu(out)
        return out

class Denoiser(nn.Module):
    def __init__(self, in_channels, out_channels, num_residual_blocks=5):
        super(Denoiser, self).__init__()
        self.conv_in = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.residual_blocks = nn.ModuleList()
        for _ in range(num_residual_blocks):
            self.residual_blocks.append(ResidualBlock(64, 64))
        self.conv_out = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.conv_in(x)
        for block in self.residual_blocks:
            out = block(out)
        out = self.conv_out(out)
        return out

class Decoder(nn.Module):
    """
    由上采样层和卷积层组成,用于融合特征并生成log difference image
    输入: (N, C_in, H, W), (N, C_in, H*2, W*2)
    输出: (N, C_out, H*2, W*2)
    """
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels*2, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )
        self.skipdown = nn.Sequential(
            nn.Conv2d(in_channels*2, out_channels, kernel_size=3, padding=1),
        )
        self.se_attention = SEAttention(out_channels)

    def forward(self, encoded, skip):
        encoded = self.up(encoded)
        skip = self.skipdown(skip)
        print("encoded:", encoded.shape)
        print("skip:", skip.shape)
        x = torch.cat([encoded, skip], 1)
        print("x",x.shape)
        out = self.conv(x)
        out = self.se_attention(out)
        return out
