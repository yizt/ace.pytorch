# -*- coding: utf-8 -*-
"""
 @File    : demo.py
 @Time    : 2020/7/3 下午4:02
 @Author  : yizuotian
 @Description    : 模型预测
"""
import argparse
import os
import sys
import string
import torch
from PIL import Image
from torchvision.transforms import transforms

from ace_utils import decode_batch
from model import ResNetEncoderDecoder

trans = transforms.Compose([
    # transforms.Resize((32, 100)),  # [h,w]
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])


def inference_image(net, alpha, image_path, device):
    image = Image.open(image_path).convert('L')
    image = trans(image)
    image = image.unsqueeze(0).to(device)  # 增加batch维
    x = net(image)
    label = decode_batch(x, alpha)
    return label


def main(args):
    device = torch.device('cuda:{}'.format(args.local_rank)
                          if args.device == 'cuda' and torch.cuda.is_available() else 'cpu')
    # 加载模型
    net = ResNetEncoderDecoder(len(args.alpha))
    net.load_state_dict(torch.load(args.weight_path, map_location='cpu')['model'])
    net.to(device)
    net.eval()
    # load image
    if args.image_dir:
        image_path_list = [os.path.join(args.image_dir, n) for n in os.listdir(args.image_dir)]
        image_path_list.sort()
        for image_path in image_path_list:
            label = inference_image(net, args.alpha, image_path, device)
            print("image_path:{},label:\n{} ".format(image_path, label))

    else:
        label = inference_image(net, args.alpha, args.image_path, device)
        print("image_path:{},label:\n{}".format(args.image_path, label))


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument("--image-path", type=str, default=None, help="test image path")
    parse.add_argument("--weight-path", type=str, default=None, help="weight path")
    parse.add_argument("--image-dir", type=str, default=None, help="test image directory")
    parse.add_argument("--device", type=str, default='cpu', help="cpu or cuda")
    parse.add_argument('--alpha', type=str, default=' ' + string.digits + string.ascii_lowercase)

    arguments = parse.parse_args(sys.argv[1:])
    main(arguments)
