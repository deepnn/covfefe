#!/usr/bin/python
# -*- coding: iso-8859-15 -*-
"""
Activation functions common API wrappers for torch frameowrk.
"""
import torch.nn as nn

# sigmoid
def sigmoid(x):
    return nn.Sigmoid()(x)

def log_sigmoid(x):
    return nn.LogSigmoid()(x)
   
# tanh
def tanh(x):
    return nn.Tanh()(x)

def hard_tanh(x, min_val=-1, max_val=1, inplace=False):
    return nn.Hardtanh(min_value=min_val, max_value=max_val, inplace=inplace)(x)

def tanh_shrink(x):
    return nn.Tanhshrink()(x)

# rectify
def relu(x, relu6=False, inplace=False):
    if relu6 == True:
        return nn.ReLU6(inplace=inplace)(x)
    else:
        return nn.ReLU(inplace=inplace)(x)

def leaky_relu(x, neg_slope=0.01, inplace=False):
    return nn.LeakyReLU(negative_slope=neg_slope, inplace=inplace)(x)

def elu(x, alpha=1.0, inplace=False):
    return nn.ELU(alpha=alpha, inplace=inplace)(x)

def prelu(x, num_param=1, init=0.25):
    return nn.PReLU(num_parameters=num_param, init=init)(x)

# soft
def softplus(x, beta=1, th=20):
    return nn.Softplus(beta=beta, threshold=th)(x)

def softmin(x):
    return nn.Softmin()(x)

def softmax(x):
    return nn.Softmax()(x)

def log_softmax(x):
    return nn.LogSoftmax()(x)

def soft_shrink(x, lam=0.5):
    return nn.Softshrink(lambd=lam)(x)

def soft_sign(x):
    return nn.Softsign()(x)

# threshold
def threshold(x, th, val, inplace=False):
    return nn.Threshold(threshold=th, value=val, inplace=inplace)(x)

# linear
def linear(x):
    return x

identity = linear 
