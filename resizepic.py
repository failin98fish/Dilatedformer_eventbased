import os
import cv2

# 设置输入文件夹路径和输出文件夹路径
input_folder = "./RealBlurSourceFiles_before"
output_folder = "./RealBlurSourceFiles"

# 创建输出文件夹（如果不存在）
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历输入文件夹中的所有文件
for filename in os.listdir(input_folder):
    # 检查文件是否为图像文件
    if filename.endswith((".jpg", ".jpeg", ".png", ".bmp")):
        # 读取图像
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)

        # 调整图像大小为224x224像素
        resized_image = cv2.resize(image, (224, 224))

        # 构建输出文件路径
        output_path = os.path.join(output_folder, filename)

        # 保存调整大小后的图像
        cv2.imwrite(output_path, resized_image)

        print(f"Resized {filename} successfully.")

print("Resize process completed.")