from __future__ import absolute_import
from __future__ import print_function

import numpy as np
from collections import OrderedDict, deque

# Some global data structures to convert between the API and frameworks
# If there are multiple inputs in the network, this will clear the globals and the other layers will contibue to push to these globals until a model is built and clear them again
first_input = True
# Layers counter
index = 0
# container for all the layers of a model
# is being cleared once a model is built
all_layers = OrderedDict()
# forward fuction call lists in order of their original calls
forwards = deque()
# outputs list, this will be used for reconstructing the calls in forwards to make the forward fuction in the model
outputs = OrderedDict()

def clear_globals():
    first_input = True
    index = 0
    all_layers.clear()
    forwards.clear()
    outputs.clear()

# use these functions to add layers so all these vars management stays here including index increament
# to add individual layers at a time
def add_layer(layer, name, args):
    # append a counter to the name and return it
    name = add_layer_acc(layer, name, args)
    index += 1
    return name

# to add a layer with activations and
# possibly batch normalization
def add_layer_acc(layer, name, args):
    name += '_{}'.format(index)
    all_layers[name] = layer
    forwards.append((layer,) + args)
    return name

# add a layer to forward computation only
# this is useful for functional activations 
# and other networks with no parameters 
def add_forward(layer, name, args)
    pass


def get_vars():
    return all_layers, forwards
# This basically iterates thru the forwards queue
# and reconstructs the forward function for the model
# TODO: move this to the model function in the framework
# this module will copy the forwards queue to the 
# call_queue of the model class
def forward(self, x):
    # do the first call on the input
    # first call is always without args
    layer = self.call_queue.popleft()[0]
    x = layer(x)
    # Now push the output tensor to outputs
    outputs[layer.name] = x
    while len(forwards) is not 0:
        func_call = self.call_queue.popleft()
        # Now, separate the function from the args
        layer = func_call[0]
        args = func_call[1:]
        # Now, look up the outputs list for i/p tensors
        args = [outputs[arg] for arg in args]
        # and call the funtion
        x = layer(*args)
        outputs[layer.name] = x

    return x
