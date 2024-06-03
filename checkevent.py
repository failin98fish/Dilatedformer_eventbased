import numpy as np
import matplotlib.pyplot as plt


def merge_data_to_image(data, image_size, time_interval):
    # 提取时间、x、y和值的列
    time = data[:, 0]
    x = data[:, 1]
    y = data[:, 2]
    value = data[:, 3]

    # 初始化图像矩阵
    image = np.zeros(image_size)

    # 遍历数据并将值合并到图像中
    for i in range(len(data)):
        if time[i] >= time_interval:
            # 显示图像
            plt.imshow(image, cmap='gray')
            plt.show()
            plt.pause(0.5)
            plt.close()
            time_interval += 0.001

        x_coord = int(x[i])
        y_coord = int(y[i])
        value_to_put = value[i]
        image[y_coord, x_coord] = value_to_put

    # 最后显示最终的图像
    plt.imshow(image, cmap='gray')
    plt.show()


# 读取.npy文件
data = np.load("mnt/disk/msc2024/runzhuw/data/test/train/others/events/scene001_000.npy")

# 图像的大小
image_size = (256,321)

# 将数据合并成图片并展示
time_interval = 0.001
merge_data_to_image(data, image_size, time_interval)