import functools

import torch
import torch.nn as nn

from base.base_model import BaseModel
from utils.util import torch_laplacian

from .networks import get_norm_layer, Chuncked_Self_Attn_FM, DenseBlock, SEBlock, AttentionUnetBackbone, \
    AutoencoderBackbone
"""
        for learning S (sharp image (APS or RGB))

        as input:
        E: events (13 channel voxel grid), as float32
        B: blur APS, [0, 1], as float32
        Bi: blur image (APS or RGB), [0, 1], as float32

        as target:
        Bi_clean: clean blur image (APS or RGB), [0, 1], as float32
        F: bi-directional optical flow, as float32
        S: sharp image (APS or RGB), [0, 1] float, as float32
"""

class DefaultModel(BaseModel):
    def __init__(self, init_dim=64, n_ev=13, grid=(8, 10), norm_type='instance', use_dropout=False, rgb=False):
        super(DefaultModel, self).__init__()

        norm_layer = get_norm_layer(norm_type)
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        self.rgb = rgb

        self.B_edge_feature_extraction = nn.Sequential(
            nn.Conv2d(1, init_dim // 4, kernel_size=7, stride=1, padding=3, bias=use_bias),
            nn.InstanceNorm2d(init_dim // 4),
            nn.ReLU(True),
            Chuncked_Self_Attn_FM(init_dim // 4, latent_dim=8, subsample=True, grid=grid),
            DenseBlock(init_dim // 4, growth_rate=32, n_blocks=3)
        )
        self.E_feature_extraction = nn.Sequential(
            nn.Conv2d(n_ev, init_dim // 4, kernel_size=7, stride=1, padding=3, bias=use_bias),
            nn.InstanceNorm2d(init_dim // 4),
            nn.ReLU(True),
            Chuncked_Self_Attn_FM(init_dim // 4, latent_dim=8, subsample=True, grid=grid),
            DenseBlock(init_dim // 4, growth_rate=32, n_blocks=3)
        )
        self.fuse1 = nn.Sequential(
            nn.Conv2d(init_dim // 2, init_dim // 2, kernel_size=1, stride=1, bias=use_bias),
            SEBlock(init_dim // 2, 8)
        )
        #optical flow, bidirectional
        self.flow_block = nn.Sequential(
            nn.Conv2d(init_dim // 2, init_dim // 2, kernel_size=1, stride=1),
            norm_layer(init_dim // 2),
            nn.ReLU(True),
            nn.Conv2d(init_dim // 2, init_dim // 4, kernel_size=3, stride=1, padding=1),
            norm_layer(init_dim // 4),
            nn.ReLU(True),
            nn.Conv2d(init_dim // 4, init_dim // 8, kernel_size=1, stride=1),
            norm_layer(init_dim // 8),
            nn.ReLU(True),
            nn.Conv2d(init_dim // 8, 4, kernel_size=1, stride=1),
            nn.ReLU(True)
        )

        if self.rgb:
            self.Bi_denoise = nn.Sequential(
                AutoencoderBackbone(3, output_nc=init_dim, n_downsampling=2, n_blocks=4, norm_type=norm_type,
                                    use_dropout=use_dropout),
                nn.Conv2d(init_dim, 3, kernel_size=1, stride=1, bias=use_bias),
                nn.Tanh()
            )
            self.Bi_feature_extraction = nn.Sequential(
                nn.Conv2d(3, init_dim // 2, kernel_size=7, stride=1, padding=3, bias=use_bias),
                norm_layer(init_dim // 2),
                nn.ReLU(True)
            )
            self.up = nn.Sequential(
                nn.ConvTranspose2d(init_dim // 2, init_dim // 2, kernel_size=3, stride=3, padding=1, output_padding=2,
                                   bias=use_bias),
                norm_layer(init_dim // 2),
                nn.ReLU(True)
            )
        else:
            self.Bi_denoise = nn.Sequential(
                AutoencoderBackbone(1, output_nc=init_dim, n_downsampling=2, n_blocks=4, norm_type=norm_type,
                                    use_dropout=use_dropout),
                nn.Conv2d(init_dim, 1, kernel_size=1, stride=1, bias=use_bias),
                nn.Tanh()
            )
            self.Bi_feature_extraction = nn.Sequential(
                nn.Conv2d(1, init_dim // 2, kernel_size=7, stride=1, padding=3, bias=use_bias),
                norm_layer(init_dim // 2),
                nn.ReLU(True)
            )

        self.fuse2 = nn.Sequential(
            nn.Conv2d(init_dim, init_dim, kernel_size=1, stride=1, bias=use_bias),
            SEBlock(init_dim, 8)
        )
        self.backbone = AttentionUnetBackbone(init_dim, output_nc=init_dim, n_downsampling=3,
                                              use_conv_to_downsample=False, norm_type=norm_type,
                                              use_dropout=use_dropout, mode='res-bottleneck')
        self.out_block = nn.Sequential(
            nn.Conv2d(init_dim, 1, kernel_size=1, stride=1, bias=use_bias),
            nn.Tanh()
        )

    def forward(self, E, B, Bi):
        B_edge_feature = self.B_edge_feature_extraction(torch_laplacian(B)) #first laplace transform B and then use B_edge extract
        E_feature = self.E_feature_extraction(E) # extract E
        motion_clues = self.fuse1(torch.cat((B_edge_feature, E_feature), dim=1))
        ''' 假设B_edge_feature的形状为(N, C1, H, W) 其中N表示批次大小 C1表示B_edge_feature的通道数 H和W分别表示高度和宽度。
        假设E_feature的形状为(N, C2, H, W) 其中C2表示E_feature的通道数。
        那么，torch.cat((B_edge_feature, E_feature), dim=1)的结果形状为(N, C1 + C2, H, W)'''
        F = self.flow_block(motion_clues)

        if self.rgb:
            motion_clues = self.up(motion_clues)

        Bi_gamma = Bi ** (1 / 2.2)
        Bi_clean = torch.clamp(self.Bi_denoise(Bi_gamma) + Bi_gamma, min=0, max=1) ** 2.2
        '''使用torch.clamp函数对这个新的张量进行裁剪，将其中小于0的元素裁剪为0，大于1的元素裁剪为1。这样可以确保张量的取值范围在0到1之间'''
        Bi_feature = self.Bi_feature_extraction(Bi_clean)
        backbone_out = self.backbone(self.fuse2(torch.cat((Bi_feature, motion_clues), dim=1)))
        S = torch.clamp(self.out_block(backbone_out) + Bi_clean, min=0, max=1)
        return F, Bi_clean, S
