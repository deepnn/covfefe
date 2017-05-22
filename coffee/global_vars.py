from __future__ import absolute_import
from __future__ import print_function

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


def clear_globals():
    first_input = True
    index = 0
    all_layers.clear()
    forwards.clear()

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
def add_forward(layer, name, args):
    pass

def get_vars():
    return all_layers, forwards

def set_first_input():
    first_input = False

def reset_first_input():
    first_input = True
