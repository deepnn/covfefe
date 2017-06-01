from __future__ import absolute_import
from __future__ import print_function

import torch.nn.init as nn

def uniform(w, a=0, b=1):
    return nn.uniform(w, a=a, b=b)

def normal(w, mean=0, std=1):
    return nn.normal(w, mean=mean, std=std)

def constant(w, val):
    return nn.constant(w, val=val)

def xavier_uniform(w, gain=1):
    return nn.xavier_uniform(w, gain=gain)

def xavier_normal(w, gain=1):
    return nn.xavier_normal(w, gain=gain)

def he_uniform(w, a=0, mode='fan_in'):
    return nn.kaiming_uniform(w, a=a, mode=mode)

def he_normal(w, a=0, mode='fan_in'):
    return nn.kaiming_normal(w, a=a, mode=mode)

def orthogonal(w, gain=1):
    return nn.orthogonal(w, gain=gain)

def sparse(w, sparsity, std=0.01):
    return nn.sparse(w, sparsity=sparsity, std=std)
