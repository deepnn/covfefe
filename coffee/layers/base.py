from collections import OrderedDict

__all__ = [
    "Layer",
    "MergeLayer",
]

class Layer(object):
    def __init__(self, shape=None, name=None):
        self.input_shape = shape
        self.input_layer = None
        
        self.name = name
    @property    
    def output_shape(self):
        shape = self.get_output_shape_for(self.input_shape)

        return shape

    def __call__(self, incoming):

        raise NotImplementedError

    def get_output_shape_for(self, input_shape):
        return input_shape

class MergeLayer(Layer):
    def __init__(self, shapes=None, name=None):
        self.input_shapes = shapes
        self.name = name

    def output_shape(self):
        shape = self.get_output_shape_for(self.input_shapes)
        return shape

    def get_output_shape_for(self, input_shapes):
        raise NotImplementedError

    def __call__(self, incomings):
        raose NotImplementedError
