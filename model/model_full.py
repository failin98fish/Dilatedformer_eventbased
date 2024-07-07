import functools
import torch
import torch.nn as nn
from .networks import SEBlock, AutoencoderBackbone, get_norm_layer, TransformerWithLocality
from base.base_model import BaseModel
from utils.util import torch_laplacian

class DefaultModel(BaseModel):
    def __init__(self, init_dim=64, n_ev=13, grid=(7,7), norm_type='instance', use_dropout=False, rgb=False):
        super(DefaultModel, self).__init__()

        norm_layer = get_norm_layer(norm_type)
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        self.rgb = rgb

        self.se_block2 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=use_bias),
            SEBlock(16, 2),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.ReLU(inplace=True),
            SEBlock(16, 2),
            nn.Conv2d(16, 1, kernel_size=1, stride=1, bias=use_bias)
        )

        self.bi_denoiser = nn.Sequential(
            AutoencoderBackbone(1, output_nc=init_dim, n_downsampling=2, n_blocks=4, norm_type=norm_type, use_dropout=use_dropout),
            nn.Conv2d(init_dim, 1, kernel_size=1, stride=1, bias=use_bias),
            nn.Tanh()
        )
        self.img_stripformer = nn.Sequential(
            TransformerWithLocality(in_channels=1, out_channels=1, dim=64, heads=8, kernel_size=3)
        )
        self.events_stripformer = nn.Sequential(
            TransformerWithLocality(in_channels=13, out_channels=1, dim=64, heads=8, kernel_size=3)
        )
        self.tanh = nn.Tanh()

    def forward(self, blurred_image, noised_b_image, events):
        blurred_code = torch_laplacian(blurred_image)
        events_code = events
        # blurred_codes = []
        # events_codes = []

        bi_gamma = noised_b_image ** (1 / 2.2)
        bi_clean = torch.clamp(self.bi_denoiser(bi_gamma) + bi_gamma, min=0, max=1) ** 2.2
        # print(bi_clean.shape)

        blurred_code = self.img_stripformer(blurred_code)
        # print("after img_stripformer: ", blurred_code.shape)
        events_code = self.events_stripformer(events_code)
        # print("after events_stripformer: ", events_code.shape)

        cated = torch.cat((bi_clean, blurred_code, events_code), dim=1)
        fused = self.se_block2(cated)
        # print(fused.shape)
        log_diff = torch.neg(fused)
        # print(log_diff.shape)
        log_diff = self.tanh(log_diff)
        sharp_image = log_diff + bi_clean
        # print(sharp_image.shape)

        return bi_clean, log_diff, sharp_image, log_diff