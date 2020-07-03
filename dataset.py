# -*- coding: utf-8 -*-
"""
 @File    : dataset.py
 @Time    : 2020/7/2 下午2:27
 @Author  : yizuotian
 @Description    :
"""
import argparse
import codecs
import os
import sys

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
        im_path = self.get_img_path(index)
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


class Synth90Dataset(BaseDataset):

    def __init__(self, synth_root, alpha=None, transforms=None,
                 target_transforms=None, mode='train', **kwargs):
        super(Synth90Dataset, self).__init__(alpha=alpha,
                                             transforms=transforms,
                                             target_transforms=target_transforms,
                                             mode=mode,
                                             **kwargs)
        self.synth_root = synth_root
        self.image_path_list = self.get_image_path_list()

    def get_image_path_list(self):
        """
        标注文件格式如下：
        ./3000/6/501_UNIVERSAL_82748.jpg 82748
        ./3000/6/500_Parameters_55458.jpg 55458
        ./3000/6/499_SKEPTICAL_71251.jpg 71251
        :return:
        """
        annotation_path = os.path.join(self.synth_root, 'annotation_{}.txt'.format(self.mode))
        with codecs.open(annotation_path, mode='r', encoding='utf-8') as f:
            lines = f.readlines()
        image_path_list = []
        for line in lines:
            img_path, _ = line.strip().split(' ')
            img_path = os.path.join(self.synth_root, img_path)
            image_path_list.append(img_path)
        return image_path_list

    def get_img_path(self, index):
        return self.image_path_list[index]

    def get_gt_text(self, index):
        image_path = self.get_img_path(index)  # eg:./3000/6/501_UNIVERSAL_82748.jpg
        image_name = os.path.basename(image_path)
        text = image_name.split('_')[1]
        return text.lower()


if __name__ == '__main__':
    import string
    from torchvision.transforms import transforms

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--syn-root', type=str, default=None)
    parser.add_argument('--alpha', type=str, default=' ' + string.digits + string.ascii_lowercase)
    args = parser.parse_args(sys.argv[1:])

    trans = transforms.Compose([
        transforms.Resize((32, 100)),  # [h,w]
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    syn = Synth90Dataset(args.syn_root, args.alpha, transforms=trans)

    for i in range(10):
        print(syn[i])
