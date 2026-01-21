import os
import random

train_path = r'E:\0-code\CD\CDofHD\laymanofSEG\data\hubei\train'
val_path = r'E:\0-code\CD\CDofHD\laymanofSEG\data\hubei\val'
test_path = r'E:\0-code\CD\CDofHD\opt_sar_seg\data\hubei\test'
ml_li = ['opt', 'sar', 'gt']

li = os.listdir(os.path.join(train_path, ml_li[0]))

random.shuffle(li)

train_li = li[:706]
val_li = li[706:]
# test_li = li[2003:]

for i in li:
    if i not in train_li:
        os.remove(os.path.join(train_path, ml_li[0], i))
        os.remove(os.path.join(train_path, ml_li[1], i))
        os.remove(os.path.join(train_path, ml_li[2], i))
    if i not in val_li:
        os.remove(os.path.join(val_path, ml_li[0], i))
        os.remove(os.path.join(val_path, ml_li[1], i))
        os.remove(os.path.join(val_path, ml_li[2], i))
    # if i not in test_li:
    #     os.remove(os.path.join(test_path, ml_li[0], i))
    #     os.remove(os.path.join(test_path, ml_li[1], i))
    #     os.remove(os.path.join(test_path, ml_li[2], i))

