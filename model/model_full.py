import functools
import torch
import torch.nn as nn
from .networks import Encoder, Denoiser, Decoder, SEBlock, Chuncked_Self_Attn_FM, AutoencoderBackbone, get_norm_layer
from base.base_model import BaseModel
from utils.util import torch_laplacian

class DefaultModel(BaseModel):
    def __init__(self, init_dim=64, n_ev=13, grid=(8, 10), norm_type='instance', use_dropout=False, rgb=False):
        super(DefaultModel, self).__init__()

        norm_layer = get_norm_layer(norm_type)
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        self.rgb = rgb
        self.encoder_blurred = nn.ModuleList([
            Encoder(1, 32),
            Encoder(32, 64),
            Encoder(64, 128),
            Encoder(128, 256),
        ])
        self.encoder_events = nn.ModuleList([
            Encoder(13, 32),
            Encoder(32, 64),
            Encoder(64, 128),
            Encoder(128, 256),
        ])
        self.se_block1 = SEBlock(512, 16)
        self.se_block2 = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.ReLU(inplace=True),
            SEBlock(16, 2),
            nn.Conv2d(16, 1, kernel_size=1, stride=1, bias=use_bias)
        )
        self.denoiser = Denoiser(512, 256)
        self.self_attn = Chuncked_Self_Attn_FM(256, latent_dim=16, subsample=True, grid=grid)
        self.decoder = nn.ModuleList([
            Decoder(256, 128),
            Decoder(128, 64),
            Decoder(64, 32),
            Decoder(32, 1),
        ])
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        self.bi_denoiser = nn.Sequential(
            AutoencoderBackbone(1, output_nc=init_dim, n_downsampling=2, n_blocks=4, norm_type=norm_type, use_dropout=use_dropout),
            nn.Conv2d(init_dim, 1, kernel_size=1, stride=1, bias=use_bias),
            nn.Tanh()
        )
        self.tanh = nn.Tanh()

    def forward(self, blurred_image, noised_b_image, events):
        blurred_code = torch_laplacian(blurred_image)
        events_code = events
        blurred_codes = []
        events_codes = []

        bi_gamma = noised_b_image ** (1 / 2.2)
        bi_clean = torch.clamp(self.bi_denoiser(bi_gamma) + bi_gamma, min=0, max=1) ** 2.2
        print(bi_clean.shape)

        for blurred_enc, events_enc in zip(self.encoder_blurred, self.encoder_events):
            blurred_code, blurred_skip = blurred_enc(blurred_code)
            events_code, events_skip = events_enc(events_code)
            blurred_codes.append(blurred_skip)
            events_codes.append(events_skip)

        code = torch.cat([blurred_code, events_code], 1)
        print("code.shape after cat", code.shape)
        code = self.se_block1(code)
        code = self.denoiser(code)
        print("code.shape", code.shape)
        code = self.self_attn(code)

        for dec, blurred_skip, events_skip in zip(self.decoder, reversed(blurred_codes), reversed(events_codes)):
            print("++++++++++++++++++++++++++++++++++++++++++++++++++")
            skips = torch.cat([blurred_skip, events_skip], 1)
            print("skips", skips.shape)
            code = dec(code, skips)
            print("code.shape", code.shape)

        print("code.shape after decode", code.shape)

        cated = torch.cat((bi_clean, code), dim=1)
        fused = self.se_block2(cated)
        print(fused.shape)
        log_diff = torch.neg(fused)
        print(log_diff.shape)
        log_diff = self.tanh(log_diff)
        sharp_image = log_diff + bi_clean
        print(sharp_image.shape)

        return bi_clean, log_diff, sharp_image, log_diff