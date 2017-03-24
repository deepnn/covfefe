from __future__ import absolute_import
from __future__ import print_function
import torch.nn as nn

# core: Data, Dropout, Reshape, Permute, RepeatVevtor, Dense

# convolution
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

# pooling

# normalization

# 
