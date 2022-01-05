from typing import Callable, Iterable, Union
import tensorflow as tf
import numpy as np
from tfcv.layers.base import Layer
from tfcv.layers.conv2d import conv_output_length

__all__ = ['MaxPooling2D', 'GlobalAveragePooling2D']

class Pooling2D(Layer):

    default_name = 'pool2d'

    def __init__(
        self, 
        pool_function: Callable,
        pool_size: Union[int, Iterable],
        strides=None,
        padding: str ='valid',
        data_format: str ='NHWC',
        trainable=False, 
        name=None):
        data_format = data_format.upper()
        assert data_format in ['NCHW', 'NHWC']
        if strides == None:
            strides = pool_size
        if isinstance(pool_size, int):
            pool_size = [pool_size, pool_size]
        else:
            pool_size = list(pool_size)
        assert min(pool_size)>0
        if isinstance(strides, int):
            strides = [strides, strides]
        else:
            strides = list(strides)
        assert min(strides)>=0
        padding = padding.upper()
        assert padding in ['VALID', 'SAME']
        self._init(locals())
        super().__init__(trainable=trainable, name=name)
    
    def call(self, inputs, training=None):
        if self.data_format == 'NHWC':
            pool_shape = [1] + self.pool_size + [1]
            strides = [1] + self.strides + [1]
        else:
            pool_shape = [1,1] + self.pool_size
            strides = [1,1] + self.strides
        outputs = self.pool_function(
            inputs,
            ksize=pool_shape,
            strides=strides,
            padding=self.padding.upper(),
            data_format=self.data_format)
        return outputs

    def _build(self, input_shape):
        self._output_specs = self.compute_output_specs(input_shape)
    
    def compute_output_specs(self, input_shape):
        if isinstance(input_shape, np.ndarray):
            input_shape = input_shape.to_list()
        else:
            input_shape = list(input_shape)
        if self.data_format == 'NCHW':
            rows = input_shape[2]
            cols = input_shape[3]
        else:
            rows = input_shape[1]
            cols = input_shape[2]
        rows = conv_output_length(rows, self.pool_size[0], self.padding,
                                         self.strides[0])
        cols = conv_output_length(cols, self.pool_size[1], self.padding,
                                         self.strides[1])
        if self.data_format == 'NCHW':
            return \
                [input_shape[0], input_shape[1], rows, cols]
        else:
            return \
                [input_shape[0], rows, cols, input_shape[3]]


class GlobalPooling2D(Layer):

    default_name = 'global_pool2d'

    def __init__(self, data_format: str, keep_dims=False, name=None):
        data_format = data_format.upper()
        assert data_format in ['NCHW', 'NHWC']
        self._init(locals())
        super(GlobalPooling2D, self).__init__(trainable=False, name=name)
    
    def _build(self, input_shape):
        self._output_specs = self.compute_output_specs(input_shape)
    
    def compute_output_specs(self, input_shape):
        if self.data_format == 'NHWC':
            if self.keep_dims:
                return [input_shape[0], 1, 1, input_shape[3]]
            else:
                return [input_shape[0], input_shape[3]]
        else:
            if self.keep_dims:
                return [input_shape[0], input_shape[1], 1, 1]
            else:
                return [input_shape[0], input_shape[1]]

class MaxPooling2D(Pooling2D):

    default_name = 'maxpool2d'

    def __init__(
        self,
        pool_size=(2, 2),
        strides=None,
        padding='valid',
        data_format='NHWC',
        name=None):
        super(MaxPooling2D, self).__init__(
            tf.nn.max_pool2d,
            pool_size=pool_size, strides=strides,
            padding=padding, data_format=data_format, name=name
        )

class GlobalAveragePooling2D(GlobalPooling2D):

    default_name = 'global_avgpool2d'

    def call(self, inputs, training=None):
        if self.data_format == 'NHWC':
            return tf.math.reduce_mean(inputs, axis=[1, 2], keepdims=self.keep_dims)
        else:
            return tf.math.reduce_mean(inputs, axis=[2, 3], keepdims=self.keep_dims)