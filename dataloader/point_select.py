import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

path = r'E:\0-code\layman\laymanofCD\dataset\data\yellowriver\gt.png'
image = np.array(Image.open(path))
image = image[20:-20,20:-20]
txt_path = r'E:\0-code\layman\laymanofCD\dataset\data\yellowriver\gt.txt'
# 获取所有0值和1值的坐标
zero_coordinates = np.argwhere(image == 0)
one_coordinates = np.argwhere(image >= 1)

zero_num = 22
one_num = 22

# 确保有足够多的0值和1值坐标可供选择
if len(zero_coordinates) >= zero_num and len(one_coordinates) >= one_num:
    # 随机选择十个0值点的坐标和二十个1值点的坐标
    random_zero_indices = np.random.choice(len(zero_coordinates), zero_num, replace=False)
    random_one_indices = np.random.choice(len(one_coordinates), one_num, replace=False)

    # 提取选择的坐标
    selected_zero_coordinates = zero_coordinates[random_zero_indices]
    selected_one_coordinates = one_coordinates[random_one_indices]

    with open(txt_path, "w") as f:
        f.write("0值点坐标：\n")
        for coordinate in selected_zero_coordinates:
            f.write(f"{coordinate[0]}, {coordinate[1]}\n")

        f.write("\n1值点坐标：\n")
        for coordinate in selected_one_coordinates:
            f.write(f"{coordinate[0]}, {coordinate[1]}\n")

    # 创建一个图像显示二值数组
    plt.imshow(image, cmap='gray')

    # 显示选定的0值点和1值点
    plt.scatter(selected_zero_coordinates[:, 1], selected_zero_coordinates[:, 0], c='r', marker='o', label='Unchanged Point')
    plt.scatter(selected_one_coordinates[:, 1], selected_one_coordinates[:, 0], c='g', marker='^', label='Changed Point')

    plt.legend()
    plt.axis('off')
    plt.show()
else:
    print("数组中0值或1值的数量不足以选择所需数量的坐标。")

