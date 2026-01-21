import numpy as np
import cv2
from skimage import transform
from scipy import ndimage
import random
import os

file_path = r"E:\0-code\CD\CDofHD\laymanofCD\dataset\3\yellowriver2\2\train"

angle = random.randint(0, 360)  # 随机选择一个角度

x_translation = random.randint(-15, 15)  # 随机选择一个x轴平移量
y_translation = random.randint(-15, 15)  # 随机选择一个y轴平移量

path_list = ['t1', 't2', 'gt']

image_list = os.listdir(os.path.join(file_path,path_list[0]))

for i in path_list:
    for j in image_list:
        image_path = os.path.join(file_path, i, j)
        # 读取tif图像
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

        # 翻转
        flipped_image = np.fliplr(image)

        # 旋转
        rotated_image = ndimage.rotate(image, angle, reshape=False)

        # 平移
        rows, cols = image.shape[:2]
        translation_matrix = np.float32([[1, 0, x_translation], [0, 1, y_translation]])
        translated_image = cv2.warpAffine(image, translation_matrix, (cols, rows))


        # 保存增强后的图像
        cv2.imwrite(image_path, image)
        cv2.imwrite(image_path[:-4] + '_flipped.png', flipped_image)
        cv2.imwrite(image_path[:-4]+'_rotated.png', rotated_image)
        cv2.imwrite(image_path[:-4] + '_translated.png', translated_image)


# # 翻转
# flipped_image = np.fliplr(image)
#
# # 旋转
# angle = random.randint(0, 360)  # 随机选择一个角度
# rotated_image = ndimage.rotate(image, angle, reshape=False)
#
# # 变形缩放
# scale_factor = random.uniform(0.8, 1.2)  # 随机选择一个缩放比例
# rescaled_image = transform.rescale(image, scale_factor, preserve_range=True)
#
# # 平移
# rows, cols = image.shape[:2]
# x_translation = random.randint(-50, 50)  # 随机选择一个x轴平移量
# y_translation = random.randint(-50, 50)  # 随机选择一个y轴平移量
# translation_matrix = np.float32([[1, 0, x_translation], [0, 1, y_translation]])
# translated_image = cv2.warpAffine(image, translation_matrix, (cols, rows))
#
#
# # 色调
# hue_shift = random.randint(-20, 20)  # 随机选择一个色调偏移量
# hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# hsv_image[:, :, 0] = (hsv_image[:, :, 0] + hue_shift) % 180
# hued_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
#
# # 对比度
# contrast_factor = random.uniform(0.5, 1.5)  # 随机选择一个对比度因子
# contrast_image = cv2.convertScaleAbs(image, alpha=contrast_factor, beta=0)
#
# # 亮度
# brightness_factor = random.randint(-50, 50)  # 随机选择一个亮度因子
# brightness_image = cv2.convertScaleAbs(image, alpha=1, beta=brightness_factor)
#
# # 保存增强后的图像
# cv2.imwrite('flipped_image.tif', flipped_image)
# cv2.imwrite('rotated_image.tif', rotated_image)
# cv2.imwrite('rescaled_image.tif', rescaled_image.astype(np.uint8))
# cv2.imwrite('translated_image.tif', translated_image)
# cv2.imwrite('hued_image.tif', hued_image)
# cv2.imwrite('contrast_image.tif', contrast_image)
# cv2.imwrite('brightness_image.tif', brightness_image)






