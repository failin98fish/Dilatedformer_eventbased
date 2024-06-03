import torch
import torch.nn.functional as F

from .loss_utils.total_variation_loss import TVLoss
from .networks import CONV3_3_IN_VGG_19
from utils.util import torch_laplacian

tv = TVLoss()


# def flow_loss(F_pred, F_gt, **kwargs):
#     l1_loss_lambda = kwargs.get('l1_loss_lambda', 1)
#     l1_loss = F.l1_loss(F_pred, F_gt) * l1_loss_lambda
#     print('flow_loss: l1_loss:', l1_loss.item())

#     tv_loss_lambda = kwargs.get('tv_loss_lambda', 1)
#     tv_loss = tv(F_pred) * tv_loss_lambda
#     print('flow_loss: tv_loss:', l1_loss.item())

#     return l1_loss + tv_loss


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


def edge_loss(pred, target):
    pred_edge = torch.abs(torch_laplacian(pred))
    target_edge = torch.abs(torch_laplacian(target))
    return F.l1_loss(pred_edge, target_edge)



def loss_full(Bi_clean_pred, Bi_clean_gt, S_pred, S_gt, code, **kwargs):
    # Lf_lambda = kwargs.get('Lf_lambda', 1)
    # Lf = flow_loss(F_pred, F_gt, **kwargs['flow_loss']) * Lf_lambda
    # print('Lf:', Lf.item())

    Lr_lambda = kwargs.get('Lr_lambda', 1)
    Lr = reconstruction_loss(S_pred, S_gt, **kwargs['reconstruction_loss']) * Lr_lambda
    print('Lr:', Lr.item())

    Ld_lambda = kwargs.get('Ld_lambda', 1)
    Ld = denoise_loss(Bi_clean_pred, Bi_clean_gt, **kwargs['denoise_loss']) * Ld_lambda
    print('Ld:', Ld.item())

    loss_log_diff = torch.mean(torch.abs(code))  # log difference的L1正则化项
    print('loss_log_diff:', 0.1*loss_log_diff)
    
    
    # tv_loss_lambda = kwargs.get('tv_loss_lambda', 1)
    # tv_loss = tv(S_pred) * tv_loss_lambda
    # print(' tv_loss:', tv_loss)

    return Ld + Lr + 0.1*loss_log_diff
