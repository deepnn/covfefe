from collections import OrderedDict

__all__ = [
    "Layer",
    "MergeLayer",
]

class Layer(object):
    def __init__(self, shape=None, name=None):
        self.shape = shape        
        self.name = name

    @property    
    def output_shape(self):
        return self.shape

    def __call__(self, inputs):

        raise NotImplementedError

    def get_output_shape_for(self, input_shape):
        return NotImplementedError

class MergeLayer(Layer):
    def __init__(self, shapes=None, name=None):
        super(MergeLayer, self).__init__(shapes, name)
        self.shape = shapes
        self.name = name

    @property
    def output_shape(self):
        return self.shape

    def get_output_shape_for(self, input_shapes):
        raise NotImplementedError

    def __call__(self, incomings):
        raose NotImplementedError
