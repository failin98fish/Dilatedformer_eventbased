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
        self.seb = SEBlock(out_channels, 8)

    def forward(self, x):
        x = self.dense(x)
        # print("x:", x.shape)
        
        down = self.down(x)
        down = self.seb(down)
        # print("down:", down.shape)
        return down, x

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
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )
        self.skipdown = nn.Sequential(
            nn.Conv2d(in_channels*2, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, encoded, skip):
        encoded = self.up(encoded)
        skip = self.skipdown(skip)
        # print("encoded:", encoded.shape)
        # print("skip:", skip.shape)
        x = torch.cat([encoded, skip], 1)
        print("x",x.shape)
        return self.conv(x)


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


class Chuncked_Self_Attn_FM(nn.Module):
    """
        in_channel -> in_channel
    """

    def __init__(self, in_channel, latent_dim=8, subsample=True, grid=(8, 8)):
        super(Chuncked_Self_Attn_FM, self).__init__()

        self.self_attn_fm = Self_Attn_FM(in_channel, latent_dim=latent_dim, subsample=subsample)
        self.grid = grid

    def forward(self, x):
        N, C, H, W = x.shape
        chunk_size_H, chunk_size_W = H // self.grid[0], W // self.grid[1]
        x_ = x.reshape(N, C, self.grid[0], chunk_size_H, self.grid[1], chunk_size_W).permute(0, 2, 4, 1, 3, 5).reshape(
            N * self.grid[0] * self.grid[1], C, chunk_size_H, chunk_size_W)
        output = self.self_attn_fm(x_).reshape(N, self.grid[0], self.grid[1], C, chunk_size_H,
                                               chunk_size_W).permute(0, 3, 1, 4, 2, 5).reshape(N, C, H, W)
        return output


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
        print("autoencoder: ")
        print(sequence)
        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        out = self.model(x)
        return out