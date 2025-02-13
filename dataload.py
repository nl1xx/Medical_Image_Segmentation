import os
import cv2
import numpy as np
import torch

import helpers
from PIL import Image
from torch.utils import data
from torch.utils.data import DataLoader


palette = [[0], [128], [255]]  # one-hot的颜色表
num_classes = 3  # 分类数

def make_dataset(root, mode, fold):
    assert mode in ['train', 'val', 'test']
    items = []
    if mode == 'train':
        img_path = os.path.join(root, 'Images')
        mask_path = os.path.join(root, 'Labels')
        data_list = [l.strip('\n') for l in open(os.path.join(root, 'train{}.txt'.format(fold))).readlines()]
        for it in data_list:
            item = (os.path.join(img_path, it), os.path.join(mask_path, it))
            items.append(item)
    elif mode == 'val':
        img_path = os.path.join(root, 'Images')
        mask_path = os.path.join(root, 'Labels')
        data_list = [l.strip('\n') for l in open(os.path.join(root, 'val{}.txt'.format(fold))).readlines()]
        for it in data_list:
            item = (os.path.join(img_path, it), os.path.join(mask_path, it))
            items.append(item)
    else:
        img_path = os.path.join(root, 'Images')
        data_list = [l.strip('\n') for l in open(os.path.join(root, 'test.txt')).readlines()]
        for it in data_list:
            item = (os.path.join(img_path, 'c0', it))
            items.append(item)
    return items


class Dataset(data.Dataset):
    def __init__(self, root, mode, fold, transform=None):
        self.images = make_dataset(root, mode, fold)
        self.palette = palette
        self.mode = mode
        if len(self.images) == 0:
            raise RuntimeError('Found 0 images, please check the data set')
        self.mode = mode
        self.transform = transform

    def __getitem__(self, index):
        img_path, mask_path = self.images[index]
        file_name = mask_path.split('\\')[-1]

        img = Image.open(img_path)
        mask = Image.open(mask_path)

        img = np.array(img)
        mask = np.array(mask)
        # Image.open读取灰度图像时shape=(H, W)而非(H, W, 1)
        img = np.expand_dims(img, axis=2)
        mask = np.expand_dims(mask, axis=2)
        mask = helpers.mask_to_onehot(mask, self.palette)
        # (H, W, C)变为(C, H, W)
        img = img.transpose([2, 0, 1])
        mask = mask.transpose([2, 0, 1])

        # 将img转换回PIL图像以应用transform
        img = Image.fromarray(img.squeeze())  # 去掉多余的维度并转换为PIL图像
        if self.transform is not None:
            img = self.transform(img)  # transform会将PIL图像转换为张量
        else:
            # 如果没有transform，手动将PIL图像转换为张量
            img = torch.tensor(np.array(img), dtype=torch.float32)

        return (img, mask), file_name

    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    def demo():
        train_path = r'./datasets/Bladder/raw_data'
        val_path = r'./datasets/Bladder/raw_data'
        test_path = r'./datasets/Bladder/test'

        train_set = Dataset(train_path, 'train', 1)
        train_loader = DataLoader(train_set, batch_size=1, shuffle=False)

        for (input, mask), file_name in train_loader:
            print(input.shape)
            print(mask.shape)
            img = helpers.array_to_img(np.expand_dims(input.squeeze(), 2))
            gt = helpers.onehot_to_mask(np.array(mask.squeeze()).transpose(1, 2, 0), palette)
            gt = helpers.array_to_img(gt)
            cv2.imshow('img GT', np.uint8(np.hstack([img, gt])))
            cv2.waitKey(1000)

    demo()
