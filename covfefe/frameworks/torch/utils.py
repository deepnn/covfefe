from __future__ import absolute_import
from __future__ import print_function

import torch.nn as nn

def clip_grad(params, max_norm, norm_type=2):
    return nn.utils.clip_grad_norm(params, max_norm, norm_type)

def parallel(module, device_ids=None, output_device=None, dim=0):
    '''
        Parallelizes the module into the device_ids and
        puts the output on the output_device
    '''

    return torch.nn.DataParallel(module, device_ids,
                                 output_devie, dim)
