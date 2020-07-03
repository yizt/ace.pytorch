# -*- coding: utf-8 -*-
"""
 @File    : train.py
 @Time    : 2019/12/4 下午7:47
 @Author  : yizuotian
 @Description    :
"""

import argparse
import os
import string
import sys
import time

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import optim
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm

import utils
from ace_utils import aggregate_cross_entropy, decode_accuracy
from dataset import Synth90Dataset
from model import ResNetEncoderDecoder


# import torchvision.transforms as transforms

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed_mode(args):
    args.distributed = False
    if args.device == 'cuda' and 'WORLD_SIZE' in os.environ:
        args.distributed = True
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.rank = int(os.environ['RANK'])
        torch.cuda.set_device(args.local_rank)

        # args.local_rank, os.environ["RANK"],os.environ['WORLD_SIZE'] 会自动赋值
        print("args.local_rank:{},RANK:{},WORLD_SIZE:{}".format(args.local_rank, os.environ["RANK"],
                                                                os.environ['WORLD_SIZE']))
        print('| distributed init (rank {}): {}'.format(
            args.rank, args.dist_url), flush=True)
        torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                             world_size=args.world_size, rank=args.rank)
        setup_for_distributed(args.rank == 0)


def train_one_epoch(model, optimizer, data_loader, device, epoch, args):
    epoch_loss = 0.0
    model.train()
    for sample in tqdm(data_loader):
        image = sample['image'].to(device)
        target = sample['target'].to(device)

        outputs = model(image)  # [B,N,C]
        loss = aggregate_cross_entropy(outputs, target)
        # 梯度更新
        model.zero_grad()
        loss.backward()
        optimizer.step()
        # 当前轮的loss
        epoch_loss += loss.item() * image.size(0)

    epoch_loss = epoch_loss / len(data_loader.dataset)
    # 打印日志,保存权重
    print('Epoch: {}/{} loss: {:03f}'.format(epoch + 1, args.epochs, epoch_loss))
    return epoch_loss


def evaluate(model, data_loader, device, epoch, args):
    model.eval()
    acc_list = []
    for sample in tqdm(data_loader):
        image = sample['image'].to(device)
        target = sample['target'].to(device)

        outputs = model(image)  # [B,N,C]
        acc = decode_accuracy(outputs, target)

        # 当前轮的精度
        acc_list.append(acc)
    # 计算最终精度
    acc_np = np.concatenate(acc_list)
    accuracy = np.mean(acc_np)
    print('Epoch: {}/{} acc: {:03f}'.format(epoch + 1, args.epochs, accuracy))
    return accuracy


def train(args):
    init_distributed_mode(args)
    print(args)
    device = torch.device(
        'cuda:{}'.format(args.local_rank) if args.device == 'cuda' and torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    # data loader
    trans = transforms.Compose([
        transforms.Resize((args.height, args.width)),  # [h,w]
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    data_set = Synth90Dataset(args.syn_root,
                              args.alpha,
                              transforms=trans,
                              target_transforms=transforms.Lambda(lambda x: torch.from_numpy(x)))
    eval_data_set = Synth90Dataset(args.syn_root,
                                   args.alpha,
                                   transforms=trans,
                                   target_transforms=transforms.Lambda(lambda x: torch.from_numpy(x)),
                                   mode='val')
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(data_set)
        eval_sampler = torch.utils.data.distributed.DistributedSampler(eval_data_set)
    else:
        train_sampler = torch.utils.data.RandomSampler(data_set)
        eval_sampler = torch.utils.data.RandomSampler(eval_data_set)
    data_loader = DataLoader(data_set, batch_size=args.batch_size, sampler=train_sampler,
                             num_workers=args.workers)
    eval_data_loader = DataLoader(eval_data_set, batch_size=args.batch_size, sampler=eval_sampler,
                                  num_workers=args.workers)
    # model
    model = ResNetEncoderDecoder(len(args.alpha))
    model = model.to(device)

    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    optimizer = optim.Adadelta(model.parameters(), weight_decay=args.weight_decay)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                          device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
        model_without_ddp = model.module
    # 加载预训练模型
    if args.init_epoch > 0:
        checkpoint = torch.load(os.path.join(args.output_dir,
                                             'ace.{:03d}.pth'.format(args.init_epoch)),
                                map_location='cpu')
        optimizer.load_state_dict(checkpoint['optimizer'])
        # lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        model_without_ddp.load_state_dict(checkpoint['model'])
    # log
    log_dir = 'log-{}'.format(time.strftime("%Y%m%d", time.localtime()))
    writer = SummaryWriter(log_dir=log_dir) if utils.is_main_process() else None

    # train
    for epoch in range(args.init_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        # 训练
        loss = train_one_epoch(model, optimizer, data_loader, device, epoch, args)
        # 评估
        acc = evaluate(model, eval_data_loader, device, epoch, args)
        # 记录日志
        utils.add_scalar_on_master(writer, 'scalar/lr', optimizer.param_groups[0]['lr'], epoch + 1)
        utils.add_scalar_on_master(writer, 'scalar/train_loss', loss, epoch + 1)
        utils.add_scalar_on_master(writer, 'scalar/accuracy', acc, epoch + 1)
        utils.add_weight_history_on_master(writer, model_without_ddp, epoch + 1)
        # 更新lr
        # lr_scheduler.step(epoch)

        # 保存模型
        if args.output_dir:
            checkpoint = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                # 'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch + 1,
                'args': args}
            utils.save_on_master(
                checkpoint,
                os.path.join(args.output_dir, 'ace.{:03d}.pth'.format(epoch + 1)))
    if utils.is_main_process():
        writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--syn-root', type=str, default=None)
    parser.add_argument('--alpha', type=str, default=' ' + string.digits + string.ascii_lowercase)
    parser.add_argument('--height', type=int, default=32, help="training image's height")
    parser.add_argument('--width', type=int, default=200, help="training image's width")
    parser.add_argument("--device", type=str, default='cpu', help="cpu or cuda")
    parser.add_argument("--batch-size", type=int, default=64, help="batch size")
    parser.add_argument("--epochs", type=int, default=50, help="epochs")
    parser.add_argument("--init-epoch", type=int, default=0, help="init epoch")
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight-decay', default=1e-5, type=float, help='weight decay (default: 0)')
    parser.add_argument("--workers", type=int, default=4, help="number of workers")
    parser.add_argument('--output-dir', default='./output', help='path where to save')

    # distributed training parameters
    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-backend', default='nccl', help='backend')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    arguments = parser.parse_args(sys.argv[1:])
    train(arguments)
