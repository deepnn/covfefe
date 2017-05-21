import numpy as np

from .. import frameworks as T
from .. import activations
from .. import global_vars as u

from .base import Layer

__all__ = ["Dense"]

class Dense(Layer):
    def __init__(self,
                 units,
                 activation=None,
                 use_bias=True,
                 init='glroot_uniform',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 name = None,
                 **kwargs):
        super(Dense, self).__init__(**kwargs)
        
        if name == None:
            name = 'Dense'
            self.name = name
        else:
            self.name = name

        self.units = units
        self.activation = activaton
        
        def __call__(self, inputs):
            # set input/output shapes
            self.input_shape = inputs.shape
            self.output_shape = self.get_output_shape_for(self.input_shape)
            self.shape = self.output_shape
            # init the layer
            layer = T.dense(self.input_shape[1],
                            self.output_shape[1],
                            bias=use_bias)
            
            # get the layers of all the inputs 
            args = (x.name for x in inputs)
            self.name = u.add_layer(layer, self.name, args)
            # depending on the activation add the activation layers as well with their arguments
            act = activations.get(self.activation)
            if act not in ['linear', None]:
                name = self.activation
                args = self.name
                u.add_layer_acc(act, name, args)

            return self

        def get_output_shape_for(self, input_shape):
            return input_shape[:1] + (self.units,)
        
        def __str__(self):
            return self.name
