import os
from PIL import Image
from torchvision.transforms import Compose, Normalize,ToTensor,Resize
import numpy as np
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

img_transform = Compose([
    Resize(256, interpolation=BICUBIC),
    ToTensor(),
    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])

other_img_transform = Compose([
    Resize(256, interpolation=BICUBIC),
])

label_transform = Compose([
    Resize(256, interpolation=Image.NEAREST),
    # ToTensor(),
])


try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

path = r'E:\code\laymanofCD\dataset\yana\train0'

li = os.listdir(os.path.join(path,'label'))

# file_li = ['label', 'DEM', 'dxqf', 'image', 'pd', 'pingmql', 'poumql', 'px', 'ql']
file_li = ['DEM','label','image']

for i in file_li:
    print(i)
    for j in li:
        img_path = os.path.join(path, i, j)
        if i == 'image':
            img = np.array(other_img_transform(Image.open(img_path).convert('RGB')))
            Image.fromarray(img).save(img_path)
        elif i == 'label':
            img = np.array(label_transform(Image.open(img_path)))
            Image.fromarray(img).save(img_path)
        else:
            img = np.array(other_img_transform(Image.open(img_path)))
            Image.fromarray(img).save(img_path)