import torch
import torch.nn as nn
import torch.nn.functional as F
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
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = nn.BatchNorm2d
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

class GlobalAvgPool(nn.Module):
    """(N,C,H,W) -> (N,C)"""

    def __init__(self):
        super(GlobalAvgPool, self).__init__()

    def forward(self, x):
        N, C, H, W = x.shape
        return x.view(N, C, -1).mean(-1)
    
class SEBlock(nn.Module):
    """(N,C,H,W) -> (N,C,H,W)"""

    def __init__(self, in_channel, r):
        super(SEBlock, self).__init__()
        self.se = nn.Sequential(
            GlobalAvgPool(),
            nn.Linear(in_channel, in_channel // r),
            nn.ReLU(inplace=True),
            nn.Linear(in_channel // r, in_channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        se_weight = self.se(x).unsqueeze(-1).unsqueeze(-1)  # (N, C, 1, 1)
        # print(se_weight.shape)
        # print((x * se_weight).shape)
        return x * se_weight  # (N, C, H, W)
    
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
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.convhalf = nn.Sequential(
            nn.Conv2d(out_channels + in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x_1 = torch.cat([x, self.conv(x)], 1)
        x = self.convhalf(x_1)
        return x
        # return self.conv(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.relu = nn.ReLU(inplace=True)
        nn.InstanceNorm2d(out_channels),
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += residual
        out = self.relu(out)
        return out



class Self_Attn_FM(nn.Module):
    """ Self attention Layer for Feature Map dimension"""

    def __init__(self, in_dim, latent_dim=8, subsample=True):
        super(Self_Attn_FM, self).__init__()
        self.channel_latent = in_dim // latent_dim
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=self.channel_latent, kernel_size=1, stride=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=self.channel_latent, kernel_size=1, stride=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=self.channel_latent, kernel_size=1, stride=1)
        self.out_conv = nn.Conv2d(in_channels=self.channel_latent, out_channels=in_dim, kernel_size=1, stride=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

        if subsample:
            self.key_conv = nn.Sequential(
                self.key_conv,
                nn.MaxPool2d(2)
            )
            self.value_conv = nn.Sequential(
                self.value_conv,
                nn.MaxPool2d(2)
            )

    def forward(self, x):
        """
            inputs :
                x : input feature maps(B x C x H x W)
            returns :
                out : self attention value + input feature
        """
        batchsize, C, height, width = x.size()
        c = self.channel_latent
        # proj_query: reshape to B x N x c, N = H x W
        proj_query = self.query_conv(x).view(batchsize, c, -1).permute(0, 2, 1)
        # proj_key: reshape to B x c x N_, N_ = H_ x W_
        proj_key = self.key_conv(x).view(batchsize, c, -1)
        # energy: B x N x N_, N = H x W, N_ = H_ x W_
        energy = torch.bmm(proj_query, proj_key)
        # attention: B x N_ x N, N = H x W, N_ = H_ x W_
        attention = self.softmax(energy).permute(0, 2, 1)
        # proj_value: B x c x N_, N_ = H_ x W_
        proj_value = self.value_conv(x).view(batchsize, c, -1)
        # attention_out: B x c x N, N = H x W
        attention_out = torch.bmm(proj_value, attention)
        # out: B x C x H x W
        out = self.out_conv(attention_out.view(batchsize, c, height, width))

        out = self.gamma * out + x
        return out

class DenseCell(nn.Module):
    def __init__(self, in_channel, growth_rate, kernel_size=3):
        super(DenseCell, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=growth_rate, kernel_size=kernel_size, stride=1,
                      padding=(kernel_size - 1) // 2, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return torch.cat((x, self.conv_block(x)), dim=1)
    

class ResBlock(nn.Module):
    """
        ResBlock using bottleneck structure
        dim -> dim
    """

    def __init__(self, dim, norm_layer, use_dropout, use_bias):
        super(ResBlock, self).__init__()

        sequence = [
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=use_bias),
            norm_layer(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=use_bias),
            norm_layer(dim),
            nn.ReLU(inplace=True)
        ]
        if use_dropout:
            sequence += [nn.Dropout(0.5)]

        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        out = x + self.model(x)
        return out

class AutoencoderBackbone(nn.Module):
    """
        Autoencoder backbone
        input_nc -> output_nc
    """

    def __init__(self, input_nc, output_nc=64, n_downsampling=2, n_blocks=6, norm_type='instance', use_dropout=False):
        super(AutoencoderBackbone, self).__init__()

        norm_layer = get_norm_layer(norm_type)
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        sequence = [
            nn.Conv2d(input_nc, output_nc, kernel_size=7, stride=1, padding=3, bias=use_bias),
            norm_layer(output_nc),
            nn.ReLU(inplace=True)
        ]

        dim = output_nc
        for i in range(n_downsampling):  # downsample the feature map
            sequence += [
                nn.Conv2d(dim, 2 * dim, kernel_size=3, stride=2, padding=1, bias=use_bias),
                norm_layer(2 * dim),
                nn.ReLU(inplace=True)
            ]
            dim *= 2

        for i in range(n_blocks):  # ResBlock
            sequence += [
                ResBlock(dim, norm_layer, use_dropout, use_bias)
            ]

        for i in range(n_downsampling):  # upsample the feature map
            sequence += [
                nn.ConvTranspose2d(dim, dim // 2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
                norm_layer(dim // 2),
                nn.ReLU(inplace=True)
            ]
            dim //= 2
        # print("autoencoder: ")
        # print(sequence)
        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        out = self.model(x)
        return out




class LocalSelfAttention(nn.Module):
    def __init__(self, channels, heads=8, kernel_size=3):
        super(LocalSelfAttention, self).__init__()
        self.heads = heads
        self.scale = (channels // heads) ** -0.5
        self.kernel_size = kernel_size

        # Initialize convolution to generate queries, keys, and values
        self.to_qkv = nn.Conv2d(channels, channels * 3, 1, bias=False)
        self.unfold = nn.Unfold(kernel_size=kernel_size, padding=kernel_size//2, stride=1)
        self.fc_out = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape

        qkv = self.to_qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=1)
        q, k, v = [self.unfold(t).view(B, self.heads, C//self.heads, self.kernel_size**2, H, W) for t in (q, k, v)]

        q = q * self.scale
        dots = torch.einsum('bhncij,bhmcij->bhnmij', q, k)  # Adjusted indices for correct einsum operation
        attn = F.softmax(dots, dim=-3)  # Softmax over the third-last dimension

        out = torch.einsum('bhnmij,bhmcij->bhncij', attn, v)
        out = out.contiguous().view(B, C, -1, H, W)
        out = out.sum(dim=2)  # Sum over the window elements
        out = self.fc_out(out) + x  # Residual connection
        return out

class DilatedConvBlock(nn.Module):
    """
    A block of dilated convolutions to expand the receptive field.
    """
    def __init__(self, in_channels, out_channels, dilation):
        super(DilatedConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))

class HDCBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation_rates):
        super(HDCBlock, self).__init__()
        self.dilated_convs = nn.ModuleList([
            DilatedConvBlock(in_channels, out_channels, dilation_rate)
            for dilation_rate in dilation_rates
        ])

    def forward(self, x):
        outputs = [conv(x) for conv in self.dilated_convs]
        return sum(outputs)

class TransformerWithLocality(nn.Module):
    """
    Transformer module incorporating locality sensitivity based on the given diagram.
    """
    def __init__(self, in_channels, out_channels, dim, heads=8, kernel_size=3):
        super(TransformerWithLocality, self).__init__()
        self.input_conv = nn.Conv2d(in_channels, dim, kernel_size=1)
        self.hdc_block1 = DilatedConvBlock(dim, dim, 1)
        self.hdc_block2 = DilatedConvBlock(dim, dim, 2)
        self.hdc_block3 = DilatedConvBlock(dim, dim, 5)
        self.attn = LocalSelfAttention(dim, heads=heads, kernel_size=kernel_size)
        self.output_conv = nn.Conv2d(dim, out_channels, kernel_size=1)  # Adjusted output convolution for concatenated features

    def forward(self, x):
        x = self.input_conv(x)

        x = self.hdc_block1(x)
        # x = self.attn(x)

        x = self.hdc_block2(x)
        # x = self.attn(x)

        x = self.hdc_block3(x)
        # x = self.attn(x)

        out = self.output_conv(x)
        return out

