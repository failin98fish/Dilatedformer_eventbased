import functools

import torch
import torch.nn as nn
from torchvision import models
import math
from einops.einops import rearrange

VGG19_FEATURES = models.vgg19(pretrained=True).features
CONV3_3_IN_VGG_19 = VGG19_FEATURES[0:15].cuda()
# VGG19_0to8 = VGG19_FEATURES[0:9].cuda()
# VGG19_9to13 = VGG19_FEATURES[9:14].cuda()
# VGG19_14to22 = VGG19_FEATURES[14:23].cuda()
# VGG19_23to31 = VGG19_FEATURES[23:32].cuda()

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


class ChannelShuffle(nn.Module):
    def __init__(self, groups=8):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        N, C, H, W = x.shape
        return x.reshape(N, self.groups, C // self.groups, H, W).transpose(1, 2).reshape(N, C, H, W)


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


class DenseBlock(nn.Module):
    """
        DenseBlock using bottleneck structure
        in_channel -> in_channel
    """

    def __init__(self, in_channel, growth_rate=32, n_blocks=3):
        super(DenseBlock, self).__init__()

        sequence = nn.ModuleList()

        dim = in_channel
        for i in range(n_blocks):
            sequence.append(DenseCell(dim, growth_rate))
            dim += growth_rate

        self.dense_cells = nn.Sequential(*sequence)
        self.fusion = nn.Conv2d(in_channels=dim, out_channels=in_channel, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        return self.fusion(self.dense_cells(x)) + x


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


class AttentionBlock(nn.Module):
    """
        attention block
        x:in_channel_x  g:in_channel_g  -->  in_channel_x
    """

    def __init__(self, in_channel_x, in_channel_g, channel_t, norm_layer, use_bias):
        # in_channel_x: input signal channels
        # in_channel_g: gating signal channels
        super(AttentionBlock, self).__init__()
        self.x_block = nn.Sequential(
            nn.Conv2d(in_channel_x, channel_t, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(channel_t)
        )

        self.g_block = nn.Sequential(
            nn.Conv2d(in_channel_g, channel_t, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(channel_t)
        )

        self.t_block = nn.Sequential(
            nn.Conv2d(channel_t, 1, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, g):
        # x: (N, in_channel_x, H, W)
        # g: (N, in_channel_g, H, W)
        x_out = self.x_block(x)  # (N, channel_t, H, W)
        g_out = self.g_block(g)  # (N, channel_t, H, W)
        t_in = self.relu(x_out + g_out)  # (N, 1, H, W)
        attention_map = self.t_block(t_in)  # (N, 1, H, W)
        return x * attention_map  # (N, in_channel_x, H, W)


class SkipAutoencoderDownsamplingBlock(nn.Module):
    """
        Autoencoder downsampling block with skip links
        in_channel -> out_channel
    """

    def __init__(self, in_channel, out_channel, norm_layer, use_dropout, use_bias, use_channel_shuffle):
        super(SkipAutoencoderDownsamplingBlock, self).__init__()

        self.projection = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1)
        if use_channel_shuffle:
            self.bottleneck = nn.Sequential(
                nn.Conv2d(out_channel, out_channel, kernel_size=1, stride=1, bias=use_bias),
                norm_layer(out_channel),
                nn.ReLU(inplace=True),
                ChannelShuffle(groups=8),
                nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=use_bias),
                norm_layer(out_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channel, out_channel, kernel_size=1, stride=1, bias=use_bias),
            )
        else:
            self.bottleneck = nn.Sequential(
                nn.Conv2d(out_channel, out_channel, kernel_size=1, stride=1, bias=use_bias),
                norm_layer(out_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=use_bias),
                norm_layer(out_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channel, out_channel, kernel_size=1, stride=1, bias=use_bias),
            )
        out_sequence = [
            norm_layer(out_channel),
            nn.ReLU(inplace=True)
        ]

        if use_dropout:
            out_sequence += [nn.Dropout(0.5)]
        out_sequence += [nn.MaxPool2d(2)]

        self.out_block = nn.Sequential(*out_sequence)

    def forward(self, x):
        x_ = self.projection(x)
        out = self.out_block(x_ + self.bottleneck(x_))
        return out


class SkipAutoencoderUpsamplingBlock(nn.Module):
    """
        Autoencoder upsampling block with skip links
        x1:in_channel1  x2:in_channel2  -->  out_channel
    """

    def __init__(self, in_channel1, in_channel2, out_channel, norm_layer, use_dropout, use_bias, use_channel_shuffle):
        super(SkipAutoencoderUpsamplingBlock, self).__init__()
        # in_channel1: channels from the signal to be upsampled
        # in_channel2: channels from skip link
        self.upsample = nn.ConvTranspose2d(in_channel1, in_channel1 // 2, kernel_size=4, stride=2, padding=1,
                                           bias=use_bias)
        self.projection = nn.Conv2d(in_channel1 // 2 + in_channel2, out_channel, kernel_size=1, stride=1)
        if use_channel_shuffle:
            self.bottleneck = nn.Sequential(
                nn.Conv2d(out_channel, out_channel, kernel_size=1, stride=1, bias=use_bias),
                norm_layer(out_channel),
                nn.ReLU(inplace=True),
                ChannelShuffle(groups=8),
                nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=use_bias),
                norm_layer(out_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channel, out_channel, kernel_size=1, stride=1, bias=use_bias),
            )
        else:
            self.bottleneck = nn.Sequential(
                nn.Conv2d(out_channel, out_channel, kernel_size=1, stride=1, bias=use_bias),
                norm_layer(out_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=use_bias),
                norm_layer(out_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channel, out_channel, kernel_size=1, stride=1, bias=use_bias),
            )
        out_sequence = [
            norm_layer(out_channel),
            nn.ReLU(inplace=True)
        ]

        if use_dropout:
            out_sequence += [nn.Dropout(0.5)]

        self.out_block = nn.Sequential(*out_sequence)

    def forward(self, x1, x2):
        # x1: the signal to be upsampled
        # x2: skip link
        upsampled_x1 = self.upsample(x1)
        x_ = self.projection(torch.cat((x2, upsampled_x1), dim=1))
        out = self.out_block(x_ + self.bottleneck(x_))
        return out


class SkipAutoencoderBackbone(nn.Module):
    """
        Autoencoder backbone with skip links
        input_nc -> output_nc
    """

    def __init__(self, input_nc, output_nc=64, n_downsampling=2, n_blocks=3, norm_type='instance', use_dropout=False,
                 use_channel_shuffle=True):
        super(SkipAutoencoderBackbone, self).__init__()

        norm_layer = get_norm_layer(norm_type)
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        self.n_downsampling = n_downsampling
        self.n_blocks = n_blocks

        self.projection = nn.Sequential(
            nn.Conv2d(input_nc, output_nc, kernel_size=7, stride=1, padding=3, bias=use_bias),
            norm_layer(output_nc),
            nn.ReLU(inplace=True)
        )
        self.in_conv = nn.Sequential(
            nn.Conv2d(output_nc, output_nc, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(output_nc),
            nn.ReLU(inplace=True)
        )
        self.out_conv = nn.Sequential(
            nn.Conv2d(2 * output_nc, output_nc, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(output_nc),
            nn.ReLU(inplace=True)
        )
        self.downsampling_blocks = nn.ModuleList()
        self.upsampling_blocks = nn.ModuleList()

        dim = output_nc
        for i in range(n_downsampling):
            self.downsampling_blocks.append(
                SkipAutoencoderDownsamplingBlock(dim, 2 * dim, norm_layer, use_dropout, use_bias, use_channel_shuffle)
            )
            dim *= 2

        dense_blocks_seq = n_blocks * [DenseBlock(dim)]
        self.dense_blocks = nn.Sequential(*dense_blocks_seq)

        for i in range(n_downsampling):
            self.upsampling_blocks.append(
                SkipAutoencoderUpsamplingBlock(dim, dim // 2, dim // 2, norm_layer, use_dropout, use_bias,
                                               use_channel_shuffle)
            )
            dim //= 2

    def forward(self, x):
        x_ = self.projection(x)
        out = self.in_conv(x_)

        skip_links = list()
        for i in range(self.n_downsampling):
            skip_links.append(out)
            out = self.downsampling_blocks[i](out)

        out = self.dense_blocks(out)

        for i in range(self.n_downsampling):
            out = self.upsampling_blocks[i](out, skip_links[-i - 1])

        out = self.out_conv(torch.cat((x_, out), dim=1))
        return out


class AttentionAutoencoderUpsamplingBlock(nn.Module):
    """
        Attention autoencoder upsampling block
        x1:in_channel1  x2:in_channel2  -->  out_channel
    """

    def __init__(self, in_channel1, in_channel2, out_channel, norm_layer, use_dropout, use_bias, use_channel_shuffle):
        super(AttentionAutoencoderUpsamplingBlock, self).__init__()
        # in_channel1: channels from the signal to be upsampled (gating signal)
        # in_channel2: channels from skip link (input signal)
        self.upsample = nn.ConvTranspose2d(in_channel1, in_channel1 // 2, kernel_size=4, stride=2, padding=1,
                                           bias=use_bias)
        self.attention = AttentionBlock(in_channel2, in_channel1 // 2, in_channel2, norm_layer, use_bias)
        self.projection = nn.Conv2d(in_channel1 // 2 + in_channel2, out_channel, kernel_size=1, stride=1)
        if use_channel_shuffle:
            self.bottleneck = nn.Sequential(
                nn.Conv2d(out_channel, out_channel, kernel_size=1, stride=1, bias=use_bias),
                norm_layer(out_channel),
                nn.ReLU(inplace=True),
                ChannelShuffle(groups=8),
                nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=use_bias),
                norm_layer(out_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channel, out_channel, kernel_size=1, stride=1, bias=use_bias),
            )
        else:
            self.bottleneck = nn.Sequential(
                nn.Conv2d(out_channel, out_channel, kernel_size=1, stride=1, bias=use_bias),
                norm_layer(out_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=use_bias),
                norm_layer(out_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channel, out_channel, kernel_size=1, stride=1, bias=use_bias),
            )
        out_sequence = [
            norm_layer(out_channel),
            nn.ReLU(inplace=True)
        ]

        if use_dropout:
            out_sequence += [nn.Dropout(0.5)]

        self.out_block = nn.Sequential(*out_sequence)

    def forward(self, x1, x2):
        # x1: the signal to be upsampled (gating signal)
        # x2: skip link (input signal)
        upsampled_x1 = self.upsample(x1)
        attentioned_x2 = self.attention(x2, upsampled_x1)
        x_ = self.projection(torch.cat((attentioned_x2, upsampled_x1), dim=1))
        out = self.out_block(x_ + self.bottleneck(x_))
        return out


class AttentionAutoencoderBackbone(nn.Module):
    """
        Attention autoencoder backbone
        input_nc -> output_nc
    """

    def __init__(self, input_nc, output_nc=64, n_downsampling=2, n_blocks=3, norm_type='instance', use_dropout=False,
                 use_channel_shuffle=True):
        super(AttentionAutoencoderBackbone, self).__init__()

        norm_layer = get_norm_layer(norm_type)
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        self.n_downsampling = n_downsampling
        self.n_blocks = n_blocks

        self.projection = nn.Sequential(
            nn.Conv2d(input_nc, output_nc, kernel_size=7, stride=1, padding=3, bias=use_bias),
            norm_layer(output_nc),
            nn.ReLU(inplace=True)
        )
        self.in_conv = nn.Sequential(
            nn.Conv2d(output_nc, output_nc, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(output_nc),
            nn.ReLU(inplace=True)
        )
        self.out_conv = nn.Sequential(
            nn.Conv2d(2 * output_nc, output_nc, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(output_nc),
            nn.ReLU(inplace=True)
        )
        self.downsampling_blocks = nn.ModuleList()
        self.upsampling_blocks = nn.ModuleList()

        dim = output_nc
        for i in range(n_downsampling):
            self.downsampling_blocks.append(
                SkipAutoencoderDownsamplingBlock(dim, 2 * dim, norm_layer, use_dropout, use_bias, use_channel_shuffle)
            )
            dim *= 2

        dense_blocks_seq = n_blocks * [DenseBlock(dim)]
        self.dense_blocks = nn.Sequential(*dense_blocks_seq)

        for i in range(n_downsampling):
            self.upsampling_blocks.append(
                AttentionAutoencoderUpsamplingBlock(dim, dim // 2, dim // 2, norm_layer, use_dropout, use_bias,
                                                    use_channel_shuffle)
            )
            dim //= 2

    def forward(self, x):
        x_ = self.projection(x)
        out = self.in_conv(x_)

        skip_links = list()
        for i in range(self.n_downsampling):
            skip_links.append(out)
            out = self.downsampling_blocks[i](out)

        out = self.dense_blocks(out)

        for i in range(self.n_downsampling):
            out = self.upsampling_blocks[i](out, skip_links[-i - 1])

        out = self.out_conv(torch.cat((x_, out), dim=1))
        return out


class UnetDoubleConvBlock(nn.Module):
    """
        Unet double Conv block
        in_channel -> out_channel
    """

    def __init__(self, in_channel, out_channel, norm_layer, use_dropout, use_bias, mode='default'):
        super(UnetDoubleConvBlock, self).__init__()

        self.mode = mode

        if self.mode == 'default':
            self.model = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=use_bias),
                norm_layer(out_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=use_bias),
                norm_layer(out_channel),
                nn.ReLU(inplace=True)
            )
            out_sequence = []
        elif self.mode == 'bottleneck':
            self.model = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, bias=use_bias),
                norm_layer(out_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=use_bias),
                norm_layer(out_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channel, out_channel, kernel_size=1, stride=1, bias=use_bias),
                norm_layer(out_channel),
                nn.ReLU(inplace=True)
            )
            out_sequence = []
        elif self.mode == 'res-bottleneck':
            self.projection = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1)
            self.bottleneck = nn.Sequential(
                nn.Conv2d(out_channel, out_channel, kernel_size=1, stride=1, bias=use_bias),
                norm_layer(out_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=use_bias),
                norm_layer(out_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channel, out_channel, kernel_size=1, stride=1, bias=use_bias),
            )
            out_sequence = [
                norm_layer(out_channel),
                nn.ReLU(inplace=True)
            ]
        else:
            raise NotImplementedError('mode [%s] is not found' % self.mode)

        if use_dropout:
            out_sequence += [nn.Dropout(0.5)]

        self.out_block = nn.Sequential(*out_sequence)

    def forward(self, x):
        if self.mode == 'res-bottleneck':
            x_ = self.projection(x)
            out = self.out_block(x_ + self.bottleneck(x_))
        else:
            out = self.out_block(self.model(x))
        return out


class UnetDownsamplingBlock(nn.Module):
    """
        Unet downsampling block
        in_channel -> out_channel
    """

    def __init__(self, in_channel, out_channel, norm_layer, use_dropout, use_bias, use_conv, mode='default'):
        super(UnetDownsamplingBlock, self).__init__()

        downsampling_layers = list()
        if use_conv:
            downsampling_layers += [
                nn.Conv2d(in_channel, in_channel, kernel_size=4, stride=2, padding=1, bias=use_bias),
                norm_layer(out_channel),
                nn.ReLU(inplace=True)
            ]
        else:
            downsampling_layers += [nn.MaxPool2d(2)]

        self.model = nn.Sequential(
            nn.Sequential(*downsampling_layers),
            UnetDoubleConvBlock(in_channel, out_channel, norm_layer, use_dropout, use_bias, mode=mode)
        )

    def forward(self, x):
        out = self.model(x)
        return out


class UnetUpsamplingBlock(nn.Module):
    """
        Unet upsampling block
        x1:in_channel1  x2:in_channel2  -->  out_channel
    """

    def __init__(self, in_channel1, in_channel2, out_channel, norm_layer, use_dropout, use_bias, mode='default'):
        super(UnetUpsamplingBlock, self).__init__()
        # in_channel1: channels from the signal to be upsampled
        # in_channel2: channels from skip link
        self.upsample = nn.ConvTranspose2d(in_channel1, in_channel1 // 2, kernel_size=4, stride=2, padding=1,
                                           bias=use_bias)
        self.double_conv = UnetDoubleConvBlock(in_channel1 // 2 + in_channel2, out_channel, norm_layer, use_dropout,
                                               use_bias, mode=mode)

    def forward(self, x1, x2):
        # x1: the signal to be upsampled
        # x2: skip link
        out = torch.cat([x2, self.upsample(x1)], dim=1)
        out = self.double_conv(out)
        return out


class UnetBackbone(nn.Module):
    """
        Unet backbone
        input_nc -> output_nc
    """

    def __init__(self, input_nc, output_nc=64, n_downsampling=4, use_conv_to_downsample=True, norm_type='instance',
                 use_dropout=False, mode='default'):
        super(UnetBackbone, self).__init__()

        self.n_downsampling = n_downsampling

        norm_layer = get_norm_layer(norm_type)
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        self.double_conv_block = UnetDoubleConvBlock(input_nc, output_nc, norm_layer, use_dropout, use_bias, mode=mode)
        self.downsampling_blocks = nn.ModuleList()
        self.upsampling_blocks = nn.ModuleList()

        dim = output_nc
        for i in range(n_downsampling):
            self.downsampling_blocks.append(
                UnetDownsamplingBlock(dim, 2 * dim, norm_layer, use_dropout, use_bias, use_conv_to_downsample,
                                      mode=mode)
            )
            dim *= 2

        for i in range(n_downsampling):
            self.upsampling_blocks.append(
                UnetUpsamplingBlock(dim, dim // 2, dim // 2, norm_layer, use_dropout, use_bias, mode=mode)
            )
            dim //= 2

    def forward(self, x):
        double_conv_block_out = self.double_conv_block(x)

        downsampling_blocks_out = list()
        downsampling_blocks_out.append(
            self.downsampling_blocks[0](double_conv_block_out)
        )
        for i in range(1, self.n_downsampling):
            downsampling_blocks_out.append(
                self.downsampling_blocks[i](downsampling_blocks_out[-1])
            )

        upsampling_blocks_out = list()
        upsampling_blocks_out.append(
            self.upsampling_blocks[0](downsampling_blocks_out[-1], downsampling_blocks_out[-2])
        )
        for i in range(1, self.n_downsampling - 1):
            upsampling_blocks_out.append(
                self.upsampling_blocks[i](upsampling_blocks_out[-1], downsampling_blocks_out[-2 - i])
            )
        upsampling_blocks_out.append(
            self.upsampling_blocks[-1](upsampling_blocks_out[-1], double_conv_block_out)
        )

        out = upsampling_blocks_out[-1]
        return out


class AttentionUnetUpsamplingBlock(nn.Module):
    """
        attention Unet upsampling block
        x1:in_channel1  x2:in_channel2  -->  out_channel
    """

    def __init__(self, in_channel1, in_channel2, out_channel, norm_layer, use_dropout, use_bias, mode='default'):
        super(AttentionUnetUpsamplingBlock, self).__init__()
        # in_channel1: channels from the signal to be upsampled (gating signal)
        # in_channel2: channels from skip link (input signal)
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(in_channel1, in_channel1 // 2, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(out_channel),
            nn.ReLU(inplace=True)
        )
        self.attention = AttentionBlock(in_channel2, in_channel1 // 2, in_channel2, norm_layer, use_bias)
        self.double_conv = UnetDoubleConvBlock(in_channel1 // 2 + in_channel2, out_channel, norm_layer, use_dropout,
                                               use_bias, mode=mode)

    def forward(self, x1, x2):
        # x1: the signal to be upsampled (gating signal)
        # x2: skip link (input signal)
        upsampled_x1 = self.upsample(x1)
        attentioned_x2 = self.attention(x2, upsampled_x1)
        out = torch.cat([attentioned_x2, upsampled_x1], dim=1)
        out = self.double_conv(out)
        return out


class AttentionUnetBackbone(nn.Module):
    """
        attention Unet backbone
        input_nc -> output_nc
    """

    def __init__(self, input_nc, output_nc=64, n_downsampling=4, use_conv_to_downsample=False, norm_type='instance',
                 use_dropout=False, mode='default'):
        super(AttentionUnetBackbone, self).__init__()

        self.n_downsampling = n_downsampling

        norm_layer = get_norm_layer(norm_type)
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        self.double_conv_block = UnetDoubleConvBlock(input_nc, output_nc, norm_layer, use_dropout, use_bias, mode=mode)
        self.downsampling_blocks = nn.ModuleList()
        self.upsampling_blocks = nn.ModuleList()

        dim = output_nc
        for i in range(n_downsampling):
            self.downsampling_blocks.append(
                UnetDownsamplingBlock(dim, 2 * dim, norm_layer, use_dropout, use_bias, use_conv_to_downsample,
                                      mode=mode)
            )
            dim *= 2

        for i in range(n_downsampling):
            self.upsampling_blocks.append(
                AttentionUnetUpsamplingBlock(dim, dim // 2, dim // 2, norm_layer, use_dropout, use_bias, mode=mode)
            )
            dim //= 2

    def forward(self, x):
        double_conv_block_out = self.double_conv_block(x)

        downsampling_blocks_out = list()
        downsampling_blocks_out.append(
            self.downsampling_blocks[0](double_conv_block_out)
        )
        for i in range(1, self.n_downsampling):
            downsampling_blocks_out.append(
                self.downsampling_blocks[i](downsampling_blocks_out[-1])
            )

        upsampling_blocks_out = list()
        upsampling_blocks_out.append(
            self.upsampling_blocks[0](downsampling_blocks_out[-1], downsampling_blocks_out[-2])
        )
        for i in range(1, self.n_downsampling - 1):
            upsampling_blocks_out.append(
                self.upsampling_blocks[i](upsampling_blocks_out[-1], downsampling_blocks_out[-2 - i])
            )
        upsampling_blocks_out.append(
            self.upsampling_blocks[-1](upsampling_blocks_out[-1], double_conv_block_out)
        )

        out = upsampling_blocks_out[-1]
        return out


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
        return x * se_weight  # (N, C, H, W)

class EventFusionNetwork(nn.Module):
    def __init__(self, bg_feature_extractor, event_feature_extractor, fusion_module, init_dim):
        """
        初始化事件融合网络。
        :param bg_feature_extractor: 背景特征提取模块,用于提取B的特征。
        :param event_feature_extractor: 事件特征提取模块,用于提取E的每个channel的特征。
        :param fusion_module: 融合模块,用于将背景特征和事件特征进行融合。
        :param init_dim: 特征融合后的维度。
        """
        super(EventFusionNetwork, self).__init__()
        self.bg_feature_extractor = bg_feature_extractor
        self.event_feature_extractor = event_feature_extractor
        self.fusion_module = fusion_module
        # 增加一个额外的特征提取模块,用于从E的每个channel提取特征
        self.motion_feature_extractor = nn.Sequential(
            nn.Conv2d(1, init_dim // 4, kernel_size=7, stride=1, padding=3),
            nn.InstanceNorm2d(init_dim // 4),
            nn.ReLU(True),
        )

    def forward(self, B, E):
        """
        网络的前向传播。
        :param B: 背景图像,shape为[batch_size, 1, height, width]
        :param E: 事件图像,shape为[batch_size, num_bins, height, width]
        :return: 融合后的特征
        """
        batch_size, num_bins, height, width = E.size()
        print("Number of event bins:", num_bins)
        print("Background image shape:", B.shape)
        
        # 提取背景图像的特征
        bg_feature = self.bg_feature_extractor(B)
        
        # 初始化融合特征为背景特征
        EB_feature = bg_feature
        print("Initial fused feature shape:", EB_feature.shape)

        # 对E中的每个时间bin进行处理
        for i in range(num_bins):
            print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            # 提取当前时间bin的事件特征
            event_feature = self.event_feature_extractor(E[:, i, :, :].unsqueeze(1))
            print("Event feature shape:", event_feature.shape)
            
            # 融合当前的背景特征和事件特征
            fusion_input = torch.cat((EB_feature, event_feature), dim=1)
            EB_feature = self.fusion_module(fusion_input)
            print("Fused feature shape:", EB_feature.shape)
            
            # 将融合后的特征作为下一次融合的输入(类似ResNet的identity映射)
            EB_feature = EB_feature + bg_feature
            print("Updated fused feature shape:", EB_feature.shape)

        return EB_feature
    
class AKConv(nn.Module):
    def __init__(self, inc, outc, num_param, stride=1, bias=None):
        super(AKConv, self).__init__()
        self.num_param = num_param
        self.stride = stride
        self.conv = nn.Sequential(nn.Conv2d(inc, outc, kernel_size=(num_param, 1), stride=(num_param, 1), bias=bias),nn.BatchNorm2d(outc),nn.SiLU())  # the conv adds the BN and SiLU to compare original Conv in YOLOv5.
        self.p_conv = nn.Conv2d(inc, 2 * num_param, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_full_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):
        # N is num_param.
        offset = self.p_conv(x)
        dtype = offset.data.type()
        N = offset.size(1) // 2
        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)

        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2) - 1), torch.clamp(q_lt[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2) - 1), torch.clamp(q_rb[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        # clip p
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2) - 1), torch.clamp(p[..., N:], 0, x.size(3) - 1)], dim=-1)

        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # resampling the features based on the modified coordinates.
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # bilinear
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        x_offset = self._reshape_x_offset(x_offset, self.num_param)
        out = self.conv(x_offset)

        return out

    # generating the inital sampled shapes for the AKConv with different sizes.
    def _get_p_n(self, N, dtype):
        base_int = round(math.sqrt(self.num_param))
        row_number = self.num_param // base_int
        mod_number = self.num_param % base_int
        p_n_x,p_n_y = torch.meshgrid(
            torch.arange(0, row_number),
            torch.arange(0,base_int))
        p_n_x = torch.flatten(p_n_x)
        p_n_y = torch.flatten(p_n_y)
        if mod_number >  0:
            mod_p_n_x,mod_p_n_y = torch.meshgrid(
                torch.arange(row_number,row_number+1),
                torch.arange(0,mod_number))

            mod_p_n_x = torch.flatten(mod_p_n_x)
            mod_p_n_y = torch.flatten(mod_p_n_y)
            p_n_x,p_n_y  = torch.cat((p_n_x,mod_p_n_x)),torch.cat((p_n_y,mod_p_n_y))
        p_n = torch.cat([p_n_x,p_n_y], 0)
        p_n = p_n.view(1, 2 * N, 1, 1).type(dtype)
        return p_n

    # no zero-padding
    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(0, h * self.stride, self.stride),
            torch.arange(0, w * self.stride, self.stride))

        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)

        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1) // 2, offset.size(2), offset.size(3)

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N] * padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    
    #  Stacking resampled features in the row direction.
    @staticmethod
    def _reshape_x_offset(x_offset, num_param):
        b, c, h, w, n = x_offset.size()
        # using Conv3d
        # x_offset = x_offset.permute(0,1,4,2,3), then Conv3d(c,c_out, kernel_size =(num_param,1,1),stride=(num_param,1,1),bias= False)
        # using 1 × 1 Conv
        # x_offset = x_offset.permute(0,1,4,2,3), then, x_offset.view(b,c×num_param,h,w)  finally, Conv2d(c×num_param,c_out, kernel_size =1,stride=1,bias= False)
        # using the column conv as follow， then, Conv2d(inc, outc, kernel_size=(num_param, 1), stride=(num_param, 1), bias=bias)
        
        x_offset = rearrange(x_offset, 'b c h w n -> b c (h n) w')
        return x_offset
