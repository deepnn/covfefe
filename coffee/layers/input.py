from collections import OrderedDict

from .base import Layer

import .utils as u #import all_layers, forwards, first_input

__all__ = ["InputLayer"]

class InputLayer(Layer):
    def __init__(self, shape, name=None, **kwargs):
        self.shape = tuple(shape)
        self.name = name

        #  the all_layers and forward lists if nor already
        if u.first_input:
            u.first_input = False
            u.all_layers.clear()
            u.forwards.clear()

    @property
    def output_shape(self):
        return self.shape
