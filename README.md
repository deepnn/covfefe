# Bunna (coffee in Amharic): A deep learning wrapper for Caffe, Theano, and TensorFlow

[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/deepnn/bunna/blob/master/LICENSE)

## Welcome to Bunna (coffee)

Bunna (coffee) is a light weight python wrapper for the three major frameworks:   [Caffe](https://github.com/BVLC/caffe), [TensorFlow](https://github.com/tensorflow/tensorflow), and [Theano](https://github.com/Theano/Theano).

Bunna (coffee) has the following advantages compared to other wrapper libraries:

- It is designed to be as transparent as TFLearn and Lasagne (which lacks simpler and standard interface) with an easy interface like keras (which lacks transparency to the underlying backends)
- It is designed to be lightweight, easy to experiment with and modular with a common scikit compatible interface.
- The interface makes it easier to convert models back and forth from the underlying frameworks using the converter library.
- It is also aims to facilitate interoperability of by enabling reuse of models trained in one of the frameworks in the others.
- with eventual goal of making it a one stop place to train deep neural networks in any framework and share trained models in a generic model zoo
- it will also maintain state-of-the-art results in popular datasets in the future
