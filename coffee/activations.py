from __future__ import absolute_import
from __future__ import print_function

import six

from . import frameworks as T

def softmax():
    return T.softmax()

def elu(alpha=1.0):
    return T.elu(alpha)

def softplus(beta=1.0):
    return T.softplus(beta)

def softsign():
    return T.soft_sign()

def relu():
    return T.relu()

def leakyrelu(slope=0.01):
    return T.leaky_relu(slope)

def tanh():
    return T.tanh()

def sigmoid():
    return T.sigmoid()

def logsigmoid():
    return T.log_sigmoid()

def get(name):
    if name is None:
        return None

    if isinstance(name, six.string_types):
        name = str(name)
        members = globals().copy()
        members.update(locals())
        return members.get(name)
