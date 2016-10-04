# coffee: A Deep Learning Wrapper for Major Frameworks

[![Documentation Status](https://readthedocs.org/projects/coffee/badge/?version=latest)](http://coffee.readthedocs.io/en/latest/?badge=latest)
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/deepnn/bunna/blob/master/LICENSE)

## Welcome to coffee

coffee (bunna in Amharic) is a light weight python wrapper for the three major frameworks:   [Caffe](https://github.com/BVLC/caffe), [TensorFlow](https://github.com/tensorflow/tensorflow), and [Theano](https://github.com/Theano/Theano).

coffee's major principle is a unifying API without sacrificing transparency to the underlying frameworks. This will make the wrapper lightweight enough to give a unified API without obscuring the powerful tools of the frameworks with the option to directly expose the power of the underlying framework directly.

coffee has the following advantages compared to other wrapper libraries:

- It is designed to be as transparent as TFLearn and Lasagne with an easy interface like keras
- It is designed to be lightweight, easy to experiment with and modular with a common scikit compatible interface.
- The interface makes it easier to convert models back and forth from the underlying frameworks using the converter library.
- It also aims to facilitate interoperability of frameworks by enabling reuse of models trained in one of the frameworks in the others.
- with eventual goal of making it a one stop place to train deep neural networks in any framework and share trained models in a generic model zoo
- it will also maintain state-of-the-art results in popular datasets in the future
