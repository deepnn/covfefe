from __future__ import absolute_import
from __future__ import print_function
import torch.nn as nn

# core: Dense, Dropout, Convolutional

# Dense (Linear)
def dense(x, in_size, out_size, bias=True):
    return nn.Linear(in_size=in_size,
                    out_size=out_size,
                    bias=bias)(x)

# Dropout
def dropout(x, p=0.5, inplace=False):
    
    #TODO: in the future some preprocessing goes here
    in_dim = x.dim()
    if in_dim == 1:
        return nn.Dropout(p=p, inplace=inplace)(x)

    elif in_dim == 2:
        return nn.Dropout2d(p=p, inplace=inplace)(x)

    elif in_dim == 3:
        return nn.Dropout3d(p=p, inplace=inplace)(x)

# convolutional
# Regular convolution
def conv(x, in_ch, out_ch, kernel_size,
            stride=1, padding=0, dilation=1, groups=1, bias=True):
    
    #TODO: in the future some preprocessing goes here
    in_dim = x.dim()
    if in_dim == 1:
        return nn.Conv1d(in_ch, out_ch, kernel_size,
                        stride=stride,
                        padding=padding,
                        dilation=dilation,
                        groups=groups,
                        bias=bias)(x)

    elif in_dim == 2:
        return nn.Conv2d(in_ch, out_ch, kernel_size,
                        stride=stride,
                        padding=padding,
                        dilation=dilation,
                        groups=groups,
                        bias=bias)(x)

    elif in_dim == 3:
        return nn.Conv3d(in_ch, out_ch, kernel_size,
                        stride=stride,
                        padding=padding,
                        dilation=dilation,
                        groups=groups,
                        bias=bias)(x)
# Transposed Concolution
def conv_transpose(x, in_ch, out_ch, kernel_size,
                      stride=1, padding=0, out_padding=0, 
                      dilation=1, groups=1, bias=True):
    
    #TODO: in the future some preprocessing goes here
    in_dim = x.dim()
    if in_dim == 1:
        return nn.ConvTranspose1d(in_ch, out_ch, kernel_size,
                        stride=stride,
                        padding=padding,
                        output_padding=out_padding,
                        dilation=dilation,
                        groups=groups,
                        bias=bias)(x)

    elif in_dim == 2:
        return nn.ConvTranspose2d(in_ch, out_ch, kernel_size,
                        stride=stride,
                        padding=padding,
                        output_padding=out_padding,
                        dilation=dilation,
                        groups=groups,
                        bias=bias)(x)

    elif in_dim == 3:
        return nn.ConvTranspose3d(in_ch, out_ch, kernel_size,
                        stride=stride,
                        padding=padding,
                        output_padding=out_padding,
                        dilation=dilation,
                        groups=groups,
                        bias=bias)(x)

# pooling
def pool(x, kernel_size, power=2, output_size=None, 
            out_ratio=None, stride=None, padding=0, 
            dilation=1, return_indices=False, ceil_mode=False, 
            mode='max', count_include_pad=True, _random_samples=None):
    
    in_dim = x.dim()
    if mode == 'max':
        if in_dim == 1:
            return nn.MaxPool1d(kernel_size=kernel_size, stride=stride, 
                        padding=padding, dilation=dilation, 
                        return_indices=return_indices, 
                        ceil_mode=ceil_mode)(x)

        elif in_dim == 2:
            return nn.MaxPool2d(kernel_size=kernel_size, stride=stride, 
                        padding=padding, dilation=dilation, 
                        return_indices=return_indices, 
                        ceil_mode=ceil_mode)(x)

        elif in_dim == 3:
            return nn.MaxPool3d(kernel_size=kernel_size, stride=stride, 
                        padding=padding, dilation=dilation, 
                        return_indices=return_indices, 
                        ceil_mode=ceil_mode)(x)

    elif mode=='ave':
        if in_dim == 1:
            return nn.AvgPool1d(kernel_size=kernel_size, stride=stride, 
                        padding=padding, ceil_mode=ceil_mode, 
                        count_include_pad=count_include_pad)(x)

        elif in_dim == 2:
            return nn.AvgPool2d(kernel_size=kernel_size, stride=stride, 
                        padding=padding, ceil_mode=ceil_mode, 
                        count_include_pad=count_include_pad)(x)

        elif in_dim == 3:
            return nn.AvgPool3d(kernel_size=kernel_size, stride=stride)(x)

    elif mode=='fractional_max':
        return nn.FractionalMaxPool2d(kernel_size=kernel_size, 
                        output_size=out_size, 
                        output_ratio=out_ratio, 
                        return_indices=return_indices, 
                        _random_samples=_random_samples)(x)

    elif mode=='power':
        return nn.LPPool2d(norm_type=power, kernel_size=kernel_size, 
                        stride=stride, ceil_mode=ceil_mode)(x)

# normalization
def batch_norm(x, num_features, eps=1e-05, momentum=0.1, affine=True):
    
    in_dim = x.dim()
    if in_dim == 1:
        return nn.BatchNorm1d(num_features=num_features, 
                        eps=eps, 
                        momentum=momentum, 
                        affine=affine)(x)

    elif in_dim == 2:
        return nn.BatchNorm2d(num_features=num_features, 
                        eps=eps, 
                        momentum=momentum, 
                        affine=affine)(x)

    elif in_dim == 3:
        return nn.BatchNorm3d(num_features=num_features, 
                        eps=eps, 
                        momentum=momentum, 
                        affine=affine)(x)
