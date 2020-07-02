# -*- coding: utf-8 -*-
"""
 @File    : dataset.py
 @Time    : 2020/7/2 下午2:27
 @Author  : yizuotian
 @Description    :
"""
import numpy as np
import torch.utils.data as data
from PIL import Image


class BaseDataset(data.Dataset):
    """
    Dataset基类
    """

    def __init__(self, alpha=None, transforms=None, target_transforms=None, mode='train', **kwargs):
        """

        :param alpha: 所有字符，首个字符是空白字符
        :param transforms:
        :param target_transforms:
        :param mode:
        :param kwargs:
        """
        super(BaseDataset, self).__init__(**kwargs)
        self.alpha = alpha
        self.transforms = transforms
        self.target_transforms = target_transforms
        self.mode = mode

    def get_img_path(self, index):
        raise NotImplementedError("not implemented yet!")

    def get_gt_text(self, index):
        raise NotImplementedError("not implemented yet!")

    def __getitem__(self, index):
        """

        :param index:
        :return img: [C,W,H]
        :return target: [num_classes] 字符出现的次数
        :return gt: GT text
        """
        im_path = self.get_gt_text(index)
        img = Image.open(im_path)
        if self.transforms:
            img = self.transforms(img)

        if self.mode == 'inference':
            return {'image': img}

        gt = self.get_gt_text(index)
        text = gt.replace(' ', '')
        indices = np.array([self.alpha.index(char) for char in text])
        label = np.zeros(len(self.alpha)).astype('float32')

        # 统计每个字符出现的次数
        for idx in indices:
            label[int(idx)] += 1  # label construction for ACE

        return {'image': img,
                'target': label,
                'gt': gt}
