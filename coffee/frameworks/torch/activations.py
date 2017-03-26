#!/usr/bin/python
# -*- coding: iso-8859-15 -*-
"""
Activation functions common API wrappers for torch frameowrk.
"""
import torch.nn as nn

# sigmoid
def sigmoid():
    return nn.Sigmoid()

def log_sigmoid():
    return nn.LogSigmoid()
   
# tanh
def tanh():
    return nn.Tanh()

def hard_tanh(min_val=-1, max_val=1, inplace=False):
    return nn.Hardtanh(min_value=min_val, max_value=max_val, inplace=inplace)

def tanh_shrink():
    return nn.Tanhshrink()

# rectify
def relu(relu6=False, inplace=False):
    if relu6 == True:
        return nn.ReLU6(inplace=inplace)
    else:
        return nn.ReLU(inplace=inplace)

def leaky_relu(neg_slope=0.01, inplace=False):
    return nn.LeakyReLU(negative_slope=neg_slope, inplace=inplace)

def elu(alpha=1.0, inplace=False):
    return nn.ELU(alpha=alpha, inplace=inplace)

def prelu(num_param=1, init=0.25):
    return nn.PReLU(num_parameters=num_param, init=init)

# soft
def softplus(beta=1, th=20):
    return nn.Softplus(beta=beta, threshold=th)

def softmin():
    return nn.Softmin()

def softmax():
    return nn.Softmax()

def log_softmax():
    return nn.LogSoftmax()

def soft_shrink(lam=0.5):
    return nn.Softshrink(lambd=lam)

def soft_sign():
    return nn.Softsign()

# threshold
def threshold(th, val, inplace=False):
    return nn.Threshold(threshold=th, value=val, inplace=inplace)

# linear
def linear(x):
    return x

identity = linear 
