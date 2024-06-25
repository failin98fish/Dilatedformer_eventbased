import functools
import torch
import torch.nn as nn
from .networks import Encoder, Denoiser, Decoder, SEBlock, AutoencoderBackbone, get_norm_layer
from base.base_model import BaseModel
from utils.util import torch_laplacian

class PatchEmbed(nn.Module):
    def __init__(self, patch_size=16, in_chans=256, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, num_layers=12, mlp_ratio=4.0, qkv_bias=True, drop_rate=0.):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(embed_dim, num_heads, int(embed_dim * mlp_ratio), activation='gelu', batch_first=True)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class ViTFusion(nn.Module):
    def __init__(self, in_chans=256, embed_dim=768, patch_size=16, num_heads=12, num_layers=12, mlp_ratio=4.0):
        super().__init__()
        self.in_chans = in_chans
        self.patch_size = patch_size
        self.patch_embed = PatchEmbed(patch_size, in_chans, embed_dim)
        self.transformer = TransformerEncoder(embed_dim, num_heads, num_layers, mlp_ratio)
        self.recover = nn.Linear(embed_dim, in_chans * patch_size ** 2)

    def forward(self, x):
        b, c, h, w = x.shape
        num_patches = (h // self.patch_size) * (w // self.patch_size)
        x = self.patch_embed(x)
        x = self.transformer(x)
        x = self.recover(x)
        x = x.view(b, num_patches, self.in_chans, self.patch_size, self.patch_size)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(b, self.in_chans, h, w)
        return x


class DefaultModel(BaseModel):
    def __init__(self, init_dim=64, n_ev=13, norm_type='instance', use_dropout=False, rgb=False):
        super(DefaultModel, self).__init__()

        norm_layer = get_norm_layer(norm_type)
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        self.rgb = rgb

        self.encoder = nn.Sequential(
            nn.Conv2d(14, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.ReLU(inplace=True)
        )

        self.se_block2 = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.ReLU(inplace=True),
            SEBlock(16, 2),
            nn.Conv2d(16, 1, kernel_size=1, stride=1, bias=use_bias)
        )

        self.vit_fusion = ViTFusion()

        self.decoder = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=use_bias),
                norm_layer(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
                norm_layer(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1, bias=use_bias),
                nn.Tanh()
            )
        ])


        self.bi_denoiser = nn.Sequential(
            AutoencoderBackbone(1, output_nc=init_dim, n_downsampling=2, n_blocks=4, norm_type=norm_type, use_dropout=use_dropout),
            nn.Conv2d(init_dim, 1, kernel_size=1, stride=1, bias=use_bias),
            nn.Tanh()
        )
        self.tanh = nn.Tanh()

    def forward(self, blurred_image, noised_b_image, events):
        blurred_code = torch_laplacian(blurred_image)
        events_code = events
        print("blurred_code ", blurred_code.shape)
        print("events_code ", events_code.shape)

        bi_gamma = noised_b_image ** (1 / 2.2)
        bi_clean = torch.clamp(self.bi_denoiser(bi_gamma) + bi_gamma, min=0, max=1) ** 2.2
        print(bi_clean.shape)

        fused_input = torch.cat((blurred_code, events_code), dim=1)
        fused_feature = self.encoder(fused_input)
        fused_feature = self.vit_fusion(fused_feature)
        
        for i, dec in enumerate(self.decoder):
            fused_feature = dec(fused_feature)
        
        fused_feature = self.tanh(fused_feature)
        print("fused_feature: ", fused_feature.shape)
        print("bi_clean: ", bi_clean.shape)

        cated = torch.cat((bi_clean, fused_feature), dim=1)
        fused = self.se_block2(cated)
        print(fused.shape)
        log_diff = torch.neg(fused)
        print(log_diff.shape) 
        log_diff = self.tanh(log_diff)
        sharp_image = log_diff + bi_clean
        print(sharp_image.shape)

        return bi_clean, log_diff, sharp_image, log_diff