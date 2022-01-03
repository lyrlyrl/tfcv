from typing import Iterable, List, Tuple, Union
import numpy as np
import tensorflow as tf

from tfcv.layers.base import Layer
from tfcv.layers.utils import need_build

def _conv_output_length(input_length, filter_size, padding, stride):
    if input_length is None:
        return None
    assert padding in ['SAME', 'VALID']
    if padding == 'SAME':
        output_length = input_length
    else:
        output_length = input_length - filter_size + 1
    return (output_length + stride - 1) // stride

class Conv2D(Layer):

    default_name = 'conv2d'
    
    def __init__(
            self, 
            filters: Union[int, float], 
            kernel_size: Union[int, Iterable],
            strides: Union[int, Iterable] = 1,
            padding: str = 'valid',
            kernel_initializer: Union[str, tf.keras.initializers.Initializer] = 'glorot_uniform',
            bias_initializer: Union[str, tf.keras.initializers.Initializer] = None,
            use_bias=False,
            data_format='NHWC',
            name=None):
        if isinstance(filters, float):
            filters = int(filters)
        if filters is not None and filters < 0:
            raise ValueError(f'Received a negative value for `filters`.'
                        f'Was expecting a positive value, got {filters}.')
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        else:
            kernel_size = list(kernel_size)
        if isinstance(strides, int):
            strides = (strides, strides)
        else:
            strides = list(strides)
        if use_bias:
            assert bias_initializer != None
            if isinstance(bias_initializer, str):
                bias_initializer = tf.keras.initializers.get(bias_initializer)
        padding = padding.upper()
        if isinstance(kernel_initializer, str):
            kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self._init(locals())
        
        super(Conv2D, self).__init__(name=name)
        
    @need_build
    def call(self, inputs, **kwargs):
        outputs = tf.nn.convolution(
            inputs, 
            self.kernel,
            self.strides,
            padding=self.padding,
            data_format=self.data_format
            )
        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias, data_format=self.data_format)
        return outputs

    def _build(self, input_shape: Union[List, Tuple, np.ndarray]):
        assert len(input_shape) == 4
        if isinstance(input_shape, np.ndarray):
            input_shape = input_shape.to_list()
        else:
            input_shape = list(input_shape)
        if self.data_format == 'NCHW':
            axis = -3
            self._output_specs = input_shape[:1]+ [
                _conv_output_length(
                    length, 
                    self.kernel_size[i], 
                    self.padding, 
                    self.strides[i])
                for i, length in enumerate(input_shape[1:-1])
                ]+ [self.filters]
        else:
            axis = -1
            self._output_specs = input_shape[:1] + [self.filters] + [
                _conv_output_length(
                    length, 
                    self.kernel_size[i], 
                    self.padding, 
                    self.strides[i])
                for i, length in enumerate(input_shape[1:-1])]

        input_channel = int(input_shape[axis])

        kernel_shape = self.kernel_size + (input_channel, self.filters)

        with tf.name_scope(self.name):
            self.kernel = tf.Variable(
                initial_value = self.kernel_initializer(
                    shape = kernel_shape,
                    dtype = tf.float32),
                trainable = True,
                name = 'kernel'
            )
            if self.use_bias:
                self.bias = tf.Variable(
                    initial_value = self.bias_initializer(
                        shape = (self.filters,),
                        dtype = tf.float32),
                    trainable = True,
                    name = 'bias'
                )
        
        