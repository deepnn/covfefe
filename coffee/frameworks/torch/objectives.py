from __future__ import absolute_import
from __future__ import print_function

import torch.nn as nn

def l1_loss(size_ave=True):
    return nn.L1Loss(size_average=size_ave)

def mse_loss(size_ave=True):
    return nn.MSELoss(size_average=size_ave)

def ce_loss(loss_weight=None, size_ave=True):
    return nn.CrossEntropyLoss(weight=loss_weight,size_average=size_ave)

def log_loss(loss_weight=None, size_ave=True, dim=2):

    if dim == 1:
        return nn.NLLLoss(weight=loss_weight,size_average=size_ave)
    elif dim == 2:
        return nn.NLLLoss2d(weight=loss_weight,size_average=size_ave)

def kldiv_loss(loss_weight=None, size_ave=True):
    return nn.KLDivLoss(weight=loss_weight,size_average=size_ave)

def bce_loss(loss_weight=None, size_ave=True):
    return nn.BCELoss(weight=loss_weight,size_average=size_ave)

def mr_loss(margin=0, size_ave=True):
    return nn.MarginRankingLoss(margin=margin,size_average=size_ave)

def he_loss(size_ave=True):
    return nn.HingeEmbeddingLoss(size_average=size_ave)

def mlm_loss(size_ave=True):
    return nn.MultiLabelMarginLoss(size_average=size_ave)

def smoothl1_loss(size_ave=True):
    return nn.SmoothL1Loss(size_average=size_ave)

def sm_loss(size_ave=True):
    return nn.SoftMarginLoss(size_average=size_ave)

def mlsm_loss(loss_weight=None, size_ave=True):
    return nn.MultiLabelSoftMarginLoss(weight=loss_weight,size_average=size_ave)

def cosem_loss(margin=0, size_ave=True):
    return nn.CosineEmbeddingLoss(margin=margin, size_average=size_ave)

def mm_loss(p=1, margin=1, loss_weight=None, size_ave=True):
    return nn.MultiMarginLoss(p=p, margin=margin,
                                weight=loss_weight,size_average=size_ave)
