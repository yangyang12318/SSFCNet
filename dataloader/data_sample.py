from PIL import Image
import os

txt_path = r'E:\0-code\CD\CDofHD\laymanofCD\dataset\3\yellowriver2\2\Point_Position.txt'
image_path = r'E:\0-code\CD\CDofHD\laymanofCD\dataset\3\data\yellowriver2\t1\yellowriver2.png'
sar_path = r'E:\0-code\CD\CDofHD\laymanofCD\dataset\3\data\yellowriver2\t2\yellowriver2.png'
label_path = r'E:\0-code\CD\CDofHD\laymanofCD\dataset\3\data\yellowriver2\gt\yellowriver2.png'
save_path = r'E:\0-code\CD\CDofHD\laymanofCD\dataset\3\yellowriver2\2'
data_name = 'yellowriver2'          #shuguang、yellowriver、tuerqi、wuhan、italy
crop_size = 256

if not os.path.exists(os.path.join(save_path, 't1')):
    os.mkdir(os.path.join(save_path, 't1'))
if not os.path.exists(os.path.join(save_path, 't2')):
    os.mkdir(os.path.join(save_path, 't2'))
if not os.path.exists(os.path.join(save_path, 'gt')):
    os.mkdir(os.path.join(save_path, 'gt'))

image = Image.open(image_path)
sar = Image.open(sar_path)
label = Image.open(label_path)

with open(txt_path,'r') as f:
    li = f.readlines()
    f.close()

    num_0 = 3
    num_1 = 3

    li_0 = li[1:1+num_0]
    li_0 = [item.strip() for item in li_0]
    li_1 = li[3+num_0:]
    li_1 = [item.strip() for item in li_1]


m = 0
n = 0

for li in [li_0, li_1]:
    for i in li:
        num_list = i.split(',')
        center_y = int(num_list[0])
        center_x = int(num_list[1])

        # 计算裁剪框的左上角和右下角坐标
        left = center_x - crop_size // 2
        top = center_y - crop_size // 2
        right = center_x + crop_size // 2
        bottom = center_y + crop_size // 2

        # 裁剪图像
        cropped_image = image.crop((left, top, right, bottom))
        cropped_sar = sar.crop((left, top, right, bottom))
        cropped_label = label.crop((left, top, right, bottom))
        # 保存裁剪后的图像
        cropped_image.save(os.path.join(save_path, 't1', data_name + str(m) + '_' + str(n) + '.png'))
        cropped_sar.save(os.path.join(save_path, 't2', data_name + str(m) + '_' + str(n) + '.png'))
        cropped_label.save(os.path.join(save_path, 'gt', data_name + str(m) + '_' + str(n) + '.png'))
        n += 1
    m += 1


