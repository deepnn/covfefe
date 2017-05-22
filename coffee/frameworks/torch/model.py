from __future__ import absolute_import
from __future__ import print_function
from collections import OrderedDict

import torch 
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
	def __init__(self, forwards, all_layers):
		super(Model, self).__init__()
		# outputs list, this will be used for reconstructing 
		# the calls in forwards to make the forward fuction in the model
		self.outputs = OrderedDict()
		self.call_queue = forwards.copy() # need to clone this 
		self.all_layers = all_layers.copy() # need to clone this

	# This basically iterates thru the forwards queue
	# and reconstructs the forward function for the model
	def forward(self, x):
		# Clear the outputs cache
		self.outputs.clear()
	    # do the first call on the input
	    # first call is always without args
	    layer = self.call_queue[0][0]
	    x = layer(x)
	    # Now push the output tensor to outputs
	    self.outputs[layer.name] = x
	    for i in range(1,len(call_queue)):
	        func_call = call_queue[i]
	        # Now, separate the function from the args
	        layer = func_call[0]
	        args = func_call[1:]
	        # Now, look up the outputs list for i/p tensors
	        args = [self.outputs[arg] for arg in args]
	        # and call the funtion
	        x = layer(*args)
	        self.outputs[layer.name] = x

	    return x

class Sequential(nn.Module):
	def __init__(self, ngpu, all_layers):
		super(Sequential, self).__init__()
		self.ngpu = ngpu
		self.net = nn.Sequential(all_layers)

	 def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output
