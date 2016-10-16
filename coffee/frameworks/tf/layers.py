"""
Layers common API wrappers for tesnsorflow frameowrk.
"""
from __future__ import absolute_import, print_function
import tflearn.layers as L

# core: Data, Dropout, Reshape, Permute, RepeatVevtor, Dense
def data(placeholder=None):
    def f(shape=None, dtype=tf.float32,
             data_preprocessing=None, data_augmentation=None,
             name='data'):
             
        T = L.input_data(shape=shape, placeholder=placeholder, dtype=dtype,
         data_preprocessing=data_preprocessing, data_augmentation=data_augmentation,
         name=name)
         
        return T
    return f
    
def dense(incoming):
    def f(n_units, activation='linear', bias=True,
            weights_init='truncated_normal', bias_init='zeros',
            regularizer=None, weight_decay=0.001, trainable=True,
            restore=True, reuse=False, scope=None,
            name='dense'):
            
        T = L.fully_connected(incoming=incoming, n_units=n_units, 
                    activation=activation, bias=bias,
                    weights_init=weights_init, bias_init=bias_init,
                    regularizer=regularizer, weight_decay=weight_decay, 
                    trainable=trainable, restore=restore, reuse=reuse, 
                    scope=scope, name=name)
        return T
    return f

def dropout(incoming):
    def f(keep_prob, name='dropout'):
        T = L.dropout(incoming=incoming, keep_prob=keep_prob, name=name)
        return T
    return f
    
def reshape(incoming):
    def f(new_shape, name='reshape'):
        T = L.reshape(incoming=incoming, new_shape=new_shape, name=name)
        return T
    return f
    

# convolution

# pooling

# normalization

# 
