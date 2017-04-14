from __future__ import absolute_import
from __future__ import print_function

import torch as T
import torch.autograd as G

def tensor(*args, dtype='float'):
    '''
        Note: all types are not implemented yet
    '''
    if dtype == 'float':
        return T.FloatTensor(args)
    elif dtype == 'double':
        return T.DoubleTensor(args)
    elif dtype == 'int':
        return T.IntTensor(args)
    elif dtype == 'long':
        return T.LongTensor(args)

def variable(*args, dtype='float', grad=True):
    '''
        returns a variable that wraps the input tensor of dtype
    '''
    return G.Variable(tensor(args, dtype), requires_grad=grad)
