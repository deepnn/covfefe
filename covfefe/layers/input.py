from collections import OrderedDict

from . import global_vars as U

from .base import Layer



__all__ = ["InputLayer"]

class InputLayer(Layer):
    '''
        This serves to transmit the shape
        It is not added as actual computational layer.
        It is just a placeholder.
    '''
    def __init__(self, shape, name=None, **kwargs):
        super(InputLayer, self).__init__(**kwargs)

        self.shape = tuple(shape)
        self.name = name

    @property
    def output_shape(self):
        return self.shape
