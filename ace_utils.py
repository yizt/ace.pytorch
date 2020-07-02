# -*- coding: utf-8 -*-
"""
 @File    : ace_utils.py
 @Time    : 2020/7/2 上午10:35
 @Author  : yizuotian
 @Description    :
"""
import numpy as np
import torch


def aggregate_cross_entropy(logits, class_ids):
    """

    :param logits: [B,H,W,C]
    :param class_ids: [B,C]
    :return loss:
    """
    b, h, w, c = logits.size()
    num_cells = h * w

    x = logits.view(b, num_cells, c)
    x = x + 1e-10
    # 空白的个数=单元格数-字符数
    class_ids[:, 0] = num_cells - torch.sum(class_ids[:, 1:], dim=1)

    # ACE Implementation (four fundamental formulas)
    x = torch.sum(x, 1)  # [B,C]
    x = x / num_cells
    class_ids = class_ids / num_cells
    loss = (-torch.sum(torch.log(x) * class_ids)) / b

    return loss


def decode_batch(x, alpha=None):
    """

    :param x: B,H,W,C
    :param alpha:
    :return:
    """
    _, indices = torch.max(x, dim=-1)  # [B,H,W]
    class_ids = indices.data.cpu().numpy()  # [B,H,W]

    if alpha:
        return np.vectorize(lambda i: alpha[i])(class_ids)

    return class_ids


def decode_accuracy(x, class_ids):
    """
    类别分布方式计算精度
    :param x: [B,H,W,C]
    :param class_ids: [B,C]
    :return acc: [B] batch中每一个是否正确
    """
    max_val, _ = torch.max(x, dim=-1, keepdim=True)  # [B,H,W,1]
    predict_cls_ids = torch.sum(torch.sum(max_val == x, dim=1), dim=1)  # [B,C]
    acc = torch.prod(class_ids == predict_cls_ids, dim=1)  # [B]
    acc = acc.cpu().numpy()
    return acc
