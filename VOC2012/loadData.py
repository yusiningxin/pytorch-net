import os

import numpy as np
import scipy.io as sio
import torch
from PIL import Image
from torch.utils import data

num_classes = 21
ignore_label = 255
root = 'data'


palette = [0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128,
           128, 128, 128, 64, 0, 0, 192, 0, 0, 64, 128, 0, 192, 128, 0, 64, 0, 128, 192, 0, 128,
           64, 128, 128, 192, 128, 128, 0, 64, 0, 128, 64, 0, 0, 192, 0, 128, 192, 0, 0, 64, 128]

zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)


def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask

#加载图片
def make_dataset(mode):
    assert mode in ['train', 'val', 'test']
    items = []
    if mode == 'train':
        img_path = os.path.join(root, 'benchmark_RELEASE', 'dataset', 'img')
        mask_path = os.path.join(root, 'benchmark_RELEASE', 'dataset', 'cls')
        data_list = [l.strip('\n') for l in open(os.path.join(
            root, 'benchmark_RELEASE', 'dataset', 'train.txt')).readlines()]
        for it in data_list:
            item = (os.path.join(img_path, it + '.jpg'), os.path.join(mask_path, it + '.mat'))
            items.append(item)
    elif mode == 'val':
        img_path = os.path.join(root, 'VOC2012', 'JPEGImages')
        mask_path = os.path.join(root, 'VOC2012', 'SegmentationClass')
        data_list = [l.strip('\n') for l in open(os.path.join(
            root,  'VOC2012', 'ImageSets', 'Segmentation', 'seg11valid.txt')).readlines()]
        for it in data_list:
            item = (os.path.join(img_path, it + '.jpg'), os.path.join(mask_path, it + '.png'))
            items.append(item)
    else:
        img_path = os.path.join(root, 'VOC2012_test', 'JPEGImages')
        data_list = [l.strip('\n') for l in open(os.path.join(
            root, 'VOC2012_test', 'ImageSets', 'Segmentation', 'test.txt')).readlines()]
        for it in data_list:
            items.append((img_path, it))
    return items


# torch.utils.data.Dataset:新创建的数据集需为其子类，其子类覆盖_len_方法，提供数据集大小；覆盖_getitem_方法，支持数据集的整数索引
class VOC(data.Dataset):
    def __init__(self, mode, joint_transform=None, sliding_crop=None, inputTransform=None, labelTransform=None):
        self.imgs = make_dataset(mode)
        if len(self.imgs) == 0:
            raise RuntimeError('Found 0 images, please check the data set')
        self.mode = mode
        self.joint_transform = joint_transform
        self.sliding_crop = sliding_crop
        self.inputTransform = inputTransform
        self.labelTransform = labelTransform

    def __getitem__(self, index):
        if self.mode == 'test':
            img_path, img_name = self.imgs[index]
            img = Image.open(os.path.join(img_path, img_name + '.jpg')).convert('RGB')
            if self.inputTransform is not None:
                img = self.inputTransform(img)
            return img_name, img

        img_path, mask_path = self.imgs[index]
        img = Image.open(img_path).convert('RGB')
        #加载训练集
        if self.mode == 'train':
            mask = sio.loadmat(mask_path)['GTcls']['Segmentation'][0][0]
            mask = Image.fromarray(mask.astype(np.uint8))
        #加载验证集
        else:
            mask = Image.open(mask_path)
        #进行变换
        if self.joint_transform is not None:
            img, mask = self.joint_transform(img, mask)

        if self.sliding_crop is not None:
            img_slices, mask_slices, slices_info = self.sliding_crop(img, mask)
            if self.inputTransform is not None:
                img_slices = [self.inputTransform(e) for e in img_slices]
            if self.labelTransform is not None:
                mask_slices = [self.labelTransform(e) for e in mask_slices]
            img, mask = torch.stack(img_slices, 0), torch.stack(mask_slices, 0)
            return img, mask, torch.LongTensor(slices_info)
        else:
            if self.inputTransform is not None:
                img = self.inputTransform(img)
            if self.labelTransform is not None:
                mask = self.labelTransform(mask)
            return img, mask

    def __len__(self):
        return len(self.imgs)
