from typing import Iterable, List, Tuple, Union
import numpy as np
import tensorflow as tf

from tfcv.config import AttrDict

from tfcv.layers.base import Layer
from tfcv.layers.utils import need_build

class Linear(Layer):

    default_name = 'linear'

    def __init__(self, units, name=None):
        self._init(locals())
        super(Linear, self).__init__(name=name)
        
    @need_build
    def call(self, inputs, **kwargs):
        return tf.matmul(inputs, self.w) + self.b
    
    def _build(self, input_shape: Union[List, Tuple, np.ndarray]):
        if isinstance(input_shape, np.ndarray):
            input_shape = input_shape.to_list()
        else:
            input_shape = list(input_shape)
        input_dim = input_shape[-1]
        input_shape[-1] = self.units
        self._output_specs = input_shape

        with tf.name_scope(self.name):
            w_init = tf.random_normal_initializer()
            self.w = tf.Variable(
                initial_value=w_init(shape=(input_dim, self.units), dtype="float32"),
                trainable=True, name='W'
            )
            b_init = tf.zeros_initializer()
            self.b = tf.Variable(
                initial_value=b_init(shape=(self.units,), dtype="float32"), trainable=True, name='b'
            )


