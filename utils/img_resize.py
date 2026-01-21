import os
from PIL import Image
import numpy as np



def resize_class_images(input_folder, output_folder, new_size=(256, 256)):
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)

        # 检查是否为图像文件
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')):
            continue

        try:
            # 打开图像
            with Image.open(input_path) as img:
                # 调整大小，使用临近采样
                resized_img = img.resize(new_size, Image.Resampling.NEAREST)

                # 保存到输出文件夹
                output_path = os.path.join(output_folder, filename)
                resized_img.save(output_path)
                print(f"Resized and saved: {output_path}")
        except Exception as e:
            print(f"Error processing file {filename}: {e}")

# 设置输入和输出文件夹路径
input_folder = r'E:\0-code\CD\CDofHD\opt_sar_seg\data\cloud\gt'  # 替换为你的输入文件夹路径
output_folder = r'E:\0-code\CD\CDofHD\opt_sar_seg\data\cloud\gt'  # 替换为你的输出文件夹路径

# 调用函数进行批量处理
resize_class_images(input_folder, output_folder)


