import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch
import time
import cv2
from IPython.display import display, clear_output

# 初始化存储历史数据的列表
loss_history = []
psnr_history = []
ssim_history = []

def plot_metrics(data_dict, epoch, max_epochs, save_path):
    global loss_history, psnr_history, ssim_history

    # 从字典中获取数据并添加到历史数据列表中
    loss_history.append(data_dict['loss'])
    psnr_history.append(data_dict['PSNR'])
    ssim_history.append(data_dict['SSIM'])

    # 清除当前图表
    plt.clf()

    def plot_subgraph(data, index, label, ylabel, color):
        plt.subplot(2, 2, index)
        plt.plot(data, label=label, color=color)
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel(ylabel)
        for i, value in enumerate(data):
            if (i + 1) % 5 == 0:
                plt.text(i, value, f'{value:.2f}')

    # 绘制 Loss 子图
    plot_subgraph(loss_history, 1, 'Loss', 'Loss', 'red')

    # 绘制 PSNR 子图
    plot_subgraph(psnr_history, 2, 'PSNR', 'PSNR', 'blue')

    # 绘制 SSIM 子图
    plot_subgraph(ssim_history, 3, 'SSIM', 'SSIM', 'green')

    # 调整子图间距
    plt.tight_layout()

    # 如果是最后一个epoch，保存图像
    if epoch == max_epochs:
        plt.savefig(save_path)

    plt.pause(0.01)
    plt.close

# 假定EB_feature是一个形状为[1, 32, 256, 320]的tensor
# EB_feature = ...

def plot_tensor_images(tensor):
    # 检查输入tensor是否有四个维度
    if tensor.ndim != 4:
        raise ValueError("Input tensor should have 4 dimensions [batch, channels, height, width]")
    
    # 获取tensor的维度
    batch_size, num_images, height, width = tensor.shape
    
    # 检查批次大小是否为1
    if batch_size != 1:
        raise ValueError("Batch size should be 1 for this function.")
    
    # 创建自定义颜色映射
    cmap = mcolors.LinearSegmentedColormap.from_list(
        'custom_gray', [(0, 'black'), (0.5, 'gray'), (1, 'white')]
    )
    
    # 创建一个新的matplotlib图形和轴
    fig, ax = plt.subplots(figsize=(8, 8))
    
    for i in range(num_images):
        # 从tensor中取出第i个图片，并且确保它是二维的
        image = tensor[0, i].cpu().detach().numpy()
        
        # 显示图片
        # 显示图片
        plt.imshow(image, cmap=cmap)
        plt.axis('off') # 关闭坐标轴
        plt.title(f'Channel {i}')
        plt.show()
        
        # 暂停0.5秒
        time.sleep(0.5)
        
        # 清除输出以显示下一张图片
        clear_output(wait=True)

def tensor_to_rgb(hsv_tensor):
    # 此函数把一个HSV张量转换为RGB张量
    # hsv_tensor: (N, 3, H, W), 第二个维度包含HSV信息
    N, _, H, W = hsv_tensor.shape
    rgb_tensor = torch.zeros_like(hsv_tensor)
    for i in range(N):
        hsv_img = hsv_tensor[i].permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
        hsv_img[:, :, 0] = (hsv_img[:, :, 0] * 360) % 360  # 色调H调整到0°～360°
        hsv_img[:, :, 1] *= 255  # 饱和度S调整到0～255
        hsv_img[:, :, 2] *= 255  # 亮度V调整到0～255
        hsv_img = hsv_img.astype(np.uint8)
        rgb_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)
        rgb_tensor[i] = torch.from_numpy(rgb_img).permute(2, 0, 1) / 255.0  # 再次转换成张量
    return rgb_tensor

def show_tensor_images(tensor):
    tensor_rgb = tensor_to_rgb(tensor)
    # 此函数显示张量中的所有图像
    # tensor: (N, 3, H, W), 第二个维度是颜色通道
    N = tensor_rgb.size(0)
    plt.figure(figsize=(20, 10))
    for i in range(N):
        plt.subplot(1, N, i + 1)
        plt.imshow(tensor_rgb[i].permute(1, 2, 0).cpu().numpy())  # 转换为(H, W, 3)用于显示
        plt.axis('off')
    plt.show()

def plot_grayscale_image(b, bi, bgt):
    # b, bi, bgt: tensors with shape [1, 1, H, W]
    images = [b.detach().squeeze().cpu().numpy(),
              bi.detach().squeeze().cpu().numpy(),
              bgt.detach().squeeze().cpu().numpy()]

    titles = ['bi', 'bipred', 'bigt']

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    for i, (image, title) in enumerate(zip(images, titles)):
        x = range(image.shape[1])
        y = image[0].flatten()
        axs[i].plot(x, y, color='black')
        axs[i].set_title(title)
        axs[i].axis('off')

    plt.show()
    plt.pause(0.01)
    plt.close()