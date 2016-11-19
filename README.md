# coffee: A Deep Learning Wrapper for Major Frameworks

[![Documentation Status](https://readthedocs.org/projects/coffee/badge/?version=latest)](http://coffee.readthedocs.io/en/latest/?badge=latest)
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/deepnn/bunna/blob/master/LICENSE)

## Welcome to coffee

coffee (bunna in Amharic) is a light weight python wrapper for the three major frameworks:   [Caffe](https://github.com/BVLC/caffe), [TensorFlow](https://github.com/tensorflow/tensorflow), and [Theano](https://github.com/Theano/Theano).

coffee's major principle is a unifying API without sacrificing transparency to the underlying frameworks. This will make the wrapper lightweight enough to give a unified API without obscuring the powerful tools of the frameworks with the option to directly expose the power of the underlying framework directly.

coffee has the following advantages compared to other wrapper libraries:

- It is designed to be as transparent as TFLearn and Lasagne and with an easy interface like keras
- It is designed to be lightweight, easy to experiment with and modular with a common scikit-like interface.
- The interface makes it easier to convert models back and forth from the underlying frameworks.
- Its input layers are like caffe (uses caffe principles interface without the pain to write complex prototxt files) data layers that supoort several data sources: lmdb, hdf5 (via h5py) folders and simple stream of numpy arrays or signle image
- It also aims to facilitate interoperability of frameworks by enabling reuse of models trained in one of the frameworks in the others.
- with eventual goal of making it a one stop place to train deep neural networks in any framework and share trained models in a generic model zoo
- it will also maintain state-of-the-art results in popular datasets in the future

## Overview
```python
# Simple model that demonstrates the simplified API (very similar interface as keras)
# But supports more frameworks as a backend and is very transparent
# No allocations of additional and unnecessary memory, no unnecessarily complicated 
# pre and post processings such as gradient clipping, unintentional internal learning rate decay
# More importantly, exposes the framework details like lasagne by allowing 
# training, validation, update, testing functions as parameters to the main trainin  and predict 
# loops

n_f = 32
ch = 1
row = col = 28
n_conv = 3
n_dense = 128
n_classes = 10

input = Input(input_shape=(ch,row,col),data_source='mnist.lmdb',batch_size=64)

# Note: there's no need to specify 1D, 2D, etc in the layers as that'd be inferred from 
# the input data shape that is specified in the input layer above

conv_1 = Convolution(ch, n_conv, n_conv, border_mode='same', activation='relu')(input)
conv_2 = Convolution(n_f, n_conv, n_conv, border_mode='same', activation='relu',
                       subsample=(2, 2))(conv_1)
conv_3 = Convolution(n_f*2, n_conv, n_conv, border_mode='same', activation='relu',
                       subsample=(2, 2))(conv_2)
conv_4 = Convolution(n_f*4, n_conv, n_conv, border_mode='same', activation='relu',
                       subsample=(2, 2))(conv_3)
flat = Flatten()(conv_4)
d_1 = Dense(n_dense, activation='relu')(flat)
d_2 = Dense(n_dense/2, activation='relu')(d_1)
o_1 = Dense(n_classes, activation='softmax')(d_2)
model = Model(inputs=[input], outputs=[o_1])
model.compile(losses=['categorical_crossentropy'], optimizers=['SGD'], loss_weights=[1.0])

model.fit(X, Y, train_func='', val_func='') # Here the internal framework could be exposed
```
You can find more examples in the example directory and the documentation at [Coffee Docs](http://coffeenet.ml/)

## Installation
Coming soon...
