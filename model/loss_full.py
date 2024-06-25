import torch
import torch.nn as nn
import torch.nn.functional as F

from .loss_utils.total_variation_loss import TVLoss
from .networks import CONV3_3_IN_VGG_19
from utils.util import torch_laplacian

tv = TVLoss()

def edge_loss(x, y, **kwargs):
    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)  # 将单通道灰度图重复为三通道
    if y.shape[1] == 1:
        y = y.repeat(1, 3, 1, 1)  # 将单通道灰度图重复为三通道
    
    k = torch.Tensor([[.05, .25, .4, .25, .05]])
    kernel = torch.matmul(k.t(), k).unsqueeze(0).repeat(3, 1, 1, 1)
    if torch.cuda.is_available():
        kernel = kernel.cuda()
    
    def conv_gauss(img):
        n_channels, _, kw, kh = kernel.shape
        img = F.pad(img, (kw // 2, kh // 2, kw // 2, kh // 2), mode='replicate')
        return F.conv2d(img, kernel, groups=n_channels)
    
    def laplacian_kernel(current):
        filtered = conv_gauss(current)  # filter
        down = filtered[:, :, ::2, ::2]  # downsample
        new_filter = torch.zeros_like(filtered)
        new_filter[:, :, ::2, ::2] = down * 4  # upsample
        filtered = conv_gauss(new_filter)  # filter
        diff = current - filtered
        return diff
    
    loss = F.l1_loss(laplacian_kernel(x), laplacian_kernel(y))
    return loss

def denoise_loss(Bi_clean_pred, Bi_clean_gt, **kwargs):
    l1_loss_lambda = kwargs.get('l1_loss_lambda', 1)
    l1_loss = F.l1_loss(Bi_clean_pred, Bi_clean_gt) * l1_loss_lambda
    print('denoise_loss: l1_loss:', l1_loss.item())

    l2_loss_lambda = kwargs.get('l2_loss_lambda', 1)
    l2_loss = F.mse_loss(Bi_clean_pred, Bi_clean_gt) * l2_loss_lambda
    print('denoise_loss: l2_loss:', l2_loss.item())

    return l1_loss + l2_loss


def reconstruction_loss(S_pred, S_gt, **kwargs):
    l2_loss_lambda = kwargs.get('l2_loss_lambda', 1)
    l2_loss = F.mse_loss(S_pred, S_gt) * l2_loss_lambda
    print('reconstruction_loss: l2_loss:', l2_loss.item())

    rgb = kwargs.get('rgb', False)
    model = CONV3_3_IN_VGG_19
    if rgb:
        S_pred_feature_map = model(S_pred)
        S_feature_map = model(S_gt).detach()  # we do not need the gradient of it
    else:
        S_pred_feature_map = model(torch.cat([S_pred] * 3, dim=1))
        S_feature_map = model(torch.cat([S_gt] * 3, dim=1)).detach()  # we do not need the gradient of it

    perceptual_loss_lambda = kwargs.get('perceptual_loss_lambda', 1)
    perceptual_loss = F.mse_loss(S_pred_feature_map, S_feature_map) * perceptual_loss_lambda
    print('reconstruction_loss: perceptual_loss:', perceptual_loss.item())

    return l2_loss + perceptual_loss



def loss_full(Bi_clean_pred, Bi_clean_gt, S_pred, S_gt, code, **kwargs):
    Lr_lambda = kwargs.get('Lr_lambda', 1)
    Lr = reconstruction_loss(S_pred, S_gt, **kwargs['reconstruction_loss']) * Lr_lambda
    print('Lr:', Lr.item())

    Ld_lambda = kwargs.get('Ld_lambda', 1)
    Ld = denoise_loss(Bi_clean_pred, Bi_clean_gt, **kwargs['denoise_loss']) * Ld_lambda
    print('Ld:', Ld.item())

    loss_log_diff = torch.mean(torch.abs(code))  # log difference的L1正则化项
    print('loss_log_diff:', 0.1*loss_log_diff)
    
    edge_loss_lambda = kwargs.get('edge_loss_lambda', 1)
    Le = edge_loss(S_pred, S_gt) * edge_loss_lambda
    print('Le:', Le.item())

    return Ld + Lr + 0.1*loss_log_diff + Le