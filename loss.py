# -*- coding: utf-8 -*-
"""
 @File    : loss.py
 @Time    : 2020/7/2 上午10:35
 @Author  : yizuotian
 @Description    :
"""
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
