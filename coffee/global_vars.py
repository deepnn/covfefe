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
