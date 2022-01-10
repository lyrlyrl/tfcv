from typing import Iterable, List, Tuple, Union
import numpy as np
import tensorflow as tf

from tfcv.layers.base import Layer
from tfcv.layers.utils import need_build

class Linear(Layer):

    default_name = 'linear'

    def __init__(
        self, 
        units, 
        kernel_initializer: Union[str, tf.keras.initializers.Initializer] = 'glorot_uniform',
        use_bias=True,
        bias_initializer: Union[str, tf.keras.initializers.Initializer] = 'zeros',
        trainable=True,
        name=None):
        if use_bias:
            assert bias_initializer != None
            if isinstance(bias_initializer, str):
                bias_initializer = tf.keras.initializers.get(bias_initializer)
        if isinstance(kernel_initializer, str):
            kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self._init(locals())
        super(Linear, self).__init__(name=name)
        
    @need_build
    def call(self, inputs, training=None):
        if self.use_bias:
            return tf.matmul(inputs, self.kernel) + self.bias
        else:
            return tf.matmul(inputs, self.kernel)
    
    def _build(self, input_shape: Union[List, Tuple, np.ndarray], training=True):
        if isinstance(input_shape, np.ndarray):
            input_shape = input_shape.to_list()
        else:
            input_shape = list(input_shape)
        input_dim = input_shape[-1]
        self._output_specs = self.compute_output_specs(input_shape)

        with tf.name_scope(self.name):
            self.kernel = tf.Variable(
                initial_value=self.kernel_initializer(shape=(input_dim, self.units), dtype="float32"),
                trainable=True, name='kernel'
            )
            if self.use_bias:
                self.bias = tf.Variable(
                    initial_value=self.bias_initializer(shape=(self.units,), dtype="float32"), trainable=True, name='bias'
                )
    def compute_output_specs(self, input_shape):
        if isinstance(input_shape, np.ndarray):
            input_shape = input_shape.to_list()
        else:
            input_shape = list(input_shape)
        input_shape[-1] = self.units
        return input_shape