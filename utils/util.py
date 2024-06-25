import os
import cv2
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
# 其中-1对应蓝色，1对应红色，中间有平滑的过渡。
def normalize(x):
    return (x - x.min()) / (x.max() - x.min()) * 2 - 1

def color_mapping(x):
    x = (x + 1) / 2  # 将x映射到[0, 1]范围内
    colors = np.zeros((x.shape[0], x.shape[1], 3))
    
    # 定义黄色和蓝色的RGB值
    yellow = np.array([1, 0, 0])
    blue = np.array([0, 0, 1])
    
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            colors[i, j] = x[i, j] * yellow + (1 - x[i, j]) * blue
    
    return colors

def create_color_image(log_diff):
    if log_diff.is_cuda:
        log_diff = log_diff.cpu()
    log_diff = normalize(log_diff)
    color_images = []
    
    for img in log_diff:
        color_img = color_mapping(img.squeeze().numpy())
        color_img = torch.from_numpy(color_img).permute(2, 0, 1)
        color_images.append(color_img)
    
    return torch.stack(color_images)

def apply_colormap(image_tensor, cmap):
    # Assuming image_tensor is normalized between -1 and 1
    if image_tensor.is_cuda:
        image_tensor = image_tensor.cpu()
    image_tensor = (image_tensor + 1) / 2  # Now between 0 and 1
    image_tensor = cmap(image_tensor.numpy())  # Apply colormap (returns rgba)
    image_tensor = torch.from_numpy(image_tensor)[..., :3]  # Drop alpha channel
    return image_tensor


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_lr_lambda(lr_lambda_tag):
    if lr_lambda_tag == 'default':
        # keep the same
        return lambda epoch: 1
    elif lr_lambda_tag == 'full':
        # 300ep
        return lambda epoch: 1
    else:
        raise NotImplementedError('lr_lambda_tag [%s] is not found' % lr_lambda_tag)


def normalize_events(events):
    # use E2VID's code to normalize the voxel grid of events
    # compute mean and stddev of the **nonzero** elements of the event tensor
    # we do not use PyTorch's default mean() and std() functions since it's faster
    # to compute it by hand than applying those funcs to a masked array
    nonzero_ev = (events != 0)
    num_nonzeros = np.sum(nonzero_ev)
    if num_nonzeros > 0:
        mean = np.sum(events) / num_nonzeros
        stddev = np.sqrt(np.sum(events ** 2) / num_nonzeros - mean ** 2)
        mask = np.float32(nonzero_ev)
        events = mask * (events - mean) / stddev
    return events


def visualize_flow_torch(flow_tensor):
    # flow_tensor: (N, 2, H, W)
    # 色调H：用角度度量，取值范围为0°～360°，从红色开始按逆时针方向计算，红色为0°，绿色为120°,蓝色为240°
    # 饱和度S：取值范围为0.0～1.0
    # 亮度V：取值范围为0.0(黑色)～1.0(白色)
    N, _, H, W = flow_tensor.shape
    visualize_tensor = torch.zeros((N, 3, H, W), dtype=torch.float32)  # (N, 3, H, W)
    for i in range(N):
        flow = flow_tensor[i].detach().cpu().numpy().transpose((1, 2, 0))
        mag, ang = cv2.cartToPolar(flow[:, :, 0], flow[:, :, 1])
        hsv = np.zeros((H, W, 3), np.uint8)  # (H, W, 3)
        hsv[:, :, 0] = ang * 180 / np.pi / 2
        hsv[:, :, 1] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        hsv[:, :, 2] = 255
        rgb = np.float32(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)) / 255  # (H, W, 3)
        visualize_tensor[i] = torch.from_numpy(rgb.transpose((2, 0, 1)))
    return visualize_tensor


@torch.jit.script
def convolve_with_kernel(img_tensor, kernel_tensor):
    # (N, C, H, W) image tensor and (h, w) kernel tensor -> (N, C, H, W) output tensor
    # kernel_tensor should be a buffer in the model to avoid runtime error when DataParallel
    # eg:
    # self.register_buffer('laplace_kernel',
    #                      torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32, requires_grad=False),
    #                      persistent=False
    #                      )
    N, C, H, W = img_tensor.shape
    h, w = kernel_tensor.shape
    return F.conv2d(
        F.pad(img_tensor.reshape(N * C, 1, H, W),
              pad=[(h - 1) // 2, (h - 1) // 2, (w - 1) // 2, (w - 1) // 2],
              mode='reflect'
              ),
        kernel_tensor.reshape(1, 1, h, w)
    ).reshape(N, C, H, W)


@torch.jit.script
def torch_laplacian(img_tensor):
    # (N, C, H, W) image tensor -> (N, C, H, W) edge tensor, the same as cv2.Laplacian
    padded = F.pad(img_tensor, pad=[1, 1, 1, 1], mode='reflect')
    return padded[:, :, 2:, 1:-1] + padded[:, :, 0:-2, 1:-1] + padded[:, :, 1:-1, 2:] + padded[:, :, 1:-1, 0:-2] - \
           4 * img_tensor


@torch.jit.script
def convert_flows_to_angle_and_length(flows):
    # (N, 2*n_flow, H, W) flows -> (N, n_flow, H, W) angle and (N, n_flow, H, W) length
    u = flows[:, 0::2, :, :]
    v = flows[:, 1::2, :, :]
    angle = torch.atan2(-v, u)
    length = torch.sqrt(u ** 2 + v ** 2)
    return angle, length

def patchify(imgs, patch_size, overlap=0):
    """
    imgs: Tensor of images, shape [B, C, H, W]
    patch_size: Tuple (pH, pW) representing the height and width of patches
    overlap: Overlap between patches
    """
    batch_size, channels, height, width = imgs.shape
    stride = patch_size - overlap
    effective_height = height - overlap
    effective_width = width - overlap
    # Number of patches in height and width
    num_patches_height = (effective_height + stride - 1) // stride
    num_patches_width = (effective_width + stride - 1) // stride
    patches = []
    
    for i in range(0, height, stride):
        for j in range(0, width, stride):
            if i + patch_size <= height and j + patch_size <= width:
                patches.append(imgs[:, :, i:i + patch_size, j:j + patch_size])
    
    patches = torch.stack(patches, dim=1)  # Shape: [B, num_patches, C, pH, pW]
    return patches

def unpatchify(patches, img_shape):
    """
    patches: Tensor of image patches, shape [B, num_patches, C, pH, pW]
    img_shape: Tuple (H, W) representing the original height and width of the image
    """
    batch_size, num_patches, channels, patch_size, _ = patches.shape
    _, _, height, width = img_shape
    stride = patch_size
    output = torch.zeros((batch_size, channels, height, width), device=patches.device)

    patch_idx = 0
    for i in range(0, height, stride):
        for j in range(0, width, stride):
            if i + patch_size <= height and j + patch_size <= width:
                output[:, :, i:i + patch_size, j:j + patch_size] += patches[:, patch_idx]
                patch_idx += 1
    
    return output
