from __future__ import absolute_import
from __future__ import print_function

from . import frameworks as T
from . import global_vars as U

from .base import Layer

class Model(Layer):
	'''
		A Wrapper Model for the actual framework Model
		It is a layer so nesting is possible.
	'''
	def __init__(self, **kwargs):
		super(Model, self).__init__(**kwargs)

		forwards, all_layers = U.get_vars()
		# The actual Model
		self.net = T.Model(forwards, all_layers)

		# Clear the globals for another model
		U.clear_globals()


	def fit():
		pass

	def train_onbatch():
		pass

	def test_onbatch():
		pass

	def predict():
		pass

	def forward():
		'''
			This performs one call on the framework model
			or simply self.net(x)
		'''
		pass

	def backward():
		''' This will accept any kind of loss
			or combination of losses to train update 
			the parameters of this model
		'''
		pass

class Sequential(Layer):
	'''
		A Wrapper for the framework Sequential model
		This serves for linear layers (no merging)
		It can be used as part of a linear segment of 
		a large complex model.
		TODO: it will have all the functionalities of Model
	'''
	def __init__(self, **kwargs):
		super(Sequential, self).__init__(**kwargs)