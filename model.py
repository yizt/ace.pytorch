# -*- coding: utf-8 -*-
"""
 @File    : model.py
 @Time    : 2020/7/2 上午10:50
 @Author  : yizuotian
 @Description    :
"""
import torch.nn.functional as F
from torch import nn
from torchvision import models


class EncoderDecoder(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.num_classes = num_classes
        # 第一层是单通道
        self.conv = nn.Conv2d(1, 64, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.bn = nn.BatchNorm2d(64)
        # 使用预训练基模型
        self.cnn = self.feature_extractor()
        # 分类
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        """

        :param x: [B,C,H,W]
        :return:
        """
        x = F.relu(self.bn(self.conv(x)), True)
        x = F.max_pool2d(x, kernel_size=(2, 2), stride=(2, 2))
        x = self.cnn(x)

        x = x.permute(0, 2, 3, 1)  # [B,C,H,W]=>[B,H,W,C]
        x = self.fc(x)
        x = F.softmax(x, dim=-1)

        return x

    @classmethod
    def feature_extractor(cls):
        return nn.Identity()


class ResNetEncoderDecoder(EncoderDecoder):
    @classmethod
    def feature_extractor(cls):
        resnet = models.resnet18(pretrained=True)
        return nn.Sequential(*list(resnet.children())[4:-2])


if __name__ == '__main__':
    import torchsummary

    net = ResNetEncoderDecoder(100)
    torchsummary.summary(net, input_size=(1, 300, 300))
