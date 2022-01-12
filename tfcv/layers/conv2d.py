from typing import Iterable, List, Tuple, Union
import numpy as np
import tensorflow as tf

from tfcv.layers.base import Layer
from tfcv.layers.utils import need_build

def conv_output_length(input_length, kernel_size, padding, stride):
    if input_length is None:
        return None
    assert padding in ['SAME', 'VALID']
    if padding == 'SAME':
        output_length = input_length
    else:
        output_length = input_length - kernel_size + 1
    return (output_length + stride - 1) // stride

def deconv_output_length(input_length,
                        kernel_size,
                        padding,
                        output_padding=None,
                        stride=0):
    assert padding in {'SAME', 'VALID'}
    if input_length is None:
        return None

    # Infer length if output padding is None, else compute the exact length
    if output_padding == None:
        if padding == 'VALID':
            length = input_length * stride + max(kernel_size - stride, 0)
        elif padding == 'SAME':
            length = input_length * stride

    else:
        if padding == 'SAME':
            pad = kernel_size // 2
        elif padding == 'VALID':
            pad = 0
        length = ((input_length - 1) * stride + kernel_size - 2 * pad +
                output_padding)
    return length
    
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
            trainable=True,
            name=None):
        if isinstance(filters, float):
            filters = int(filters)
        if filters != None and filters < 0:
            raise ValueError(f'Received a negative value for `filters`.'
                        f'Was expecting a positive value, got {filters}.')
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size, kernel_size]
        else:
            kernel_size = list(kernel_size)
        if isinstance(strides, int):
            strides = [strides, strides]
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

    def _build(self, input_shape: Union[List, Tuple, np.ndarray], training=None):
        if isinstance(input_shape, np.ndarray):
            input_shape = input_shape.to_list()
        else:
            input_shape = list(input_shape)
        self._output_specs = self.compute_output_specs(input_shape)
        if self.data_format == 'NHWC':
            axis = -1
        else:
            axis = -3

        input_channel = int(input_shape[axis])

        kernel_shape = self.kernel_size + [input_channel, self.filters]

        with tf.name_scope(self.name):
            self.kernel = self.add_weight(
                name = 'kernel',
                shape = kernel_shape,
                dtype = tf.float32,
                trainable = True,
                initializer = self.kernel_initializer
            )

            if self.use_bias:
                self.bias = self.add_weight(
                    name = 'bias',
                    shape = (self.filters,),
                    dtype = tf.float32,
                    trainable = True,
                    initializer = self.bias_initializer
                )
                
    def compute_output_specs(self, input_shape):
        if isinstance(input_shape, np.ndarray):
            input_shape = input_shape.to_list()
        else:
            input_shape = list(input_shape)
        assert len(input_shape) == 4
        if self.data_format == 'NHWC':
            return input_shape[:1]+ [
                conv_output_length(
                    length, 
                    self.kernel_size[i], 
                    self.padding, 
                    self.strides[i])
                for i, length in enumerate(input_shape[1:-1])
                ]+ [self.filters]
        else:
            return input_shape[:1] + [self.filters] + [
                conv_output_length(
                    length, 
                    self.kernel_size[i], 
                    self.padding, 
                    self.strides[i])
                for i, length in enumerate(input_shape[2:])]

class Conv2DTranspose(Conv2D):

    default_name = 'deconv2d'

    def __init__(
            self, 
            filters: Union[int, float], 
            kernel_size: Union[int, Iterable],
            strides: Union[int, Iterable] = 1,
            output_padding = None,
            padding: str = 'valid',
            kernel_initializer: Union[str, tf.keras.initializers.Initializer] = 'glorot_uniform',
            bias_initializer: Union[str, tf.keras.initializers.Initializer] = None,
            use_bias=False,
            data_format='NHWC',
            trainable=True,
            name=None):
        super(Conv2DTranspose, self).__init__(
            filters=filters, 
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            use_bias=use_bias,
            data_format=data_format,
            trainable=trainable,
            name=name
        )
        if output_padding != None:
            if isinstance(output_padding, int):
                output_padding = [output_padding, output_padding]
            else:
                output_padding = list(output_padding)
            for stride, out_pad in zip(strides, output_padding):
                if out_pad >= stride:
                    raise ValueError('Stride ' + str(strides) + ' must be '
                                'greater than output padding ' +
                                str(output_padding))
        self.output_padding = output_padding

    def _build(self, input_shape: Union[List, Tuple, np.ndarray], training=None):
        self._output_specs = self.compute_output_specs(input_shape)
        if self.data_format == 'NHWC':
            axis = -1
        else:
            axis = -3
        input_channel = int(input_shape[axis])
        kernel_shape = self.kernel_size + [self.filters, input_channel]
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

    def compute_output_specs(self, input_shape):
        if isinstance(input_shape, np.ndarray):
            input_shape = input_shape.to_list()
        else:
            input_shape = list(input_shape)
        assert len(input_shape) == 4
        if self.data_format == 'NHWC':
            return input_shape[:1]+ [
                deconv_output_length(
                    length, 
                    self.kernel_size[i], 
                    self.padding, 
                    self.strides[i])
                for i, length in enumerate(input_shape[1:-1])
                ]+ [self.filters]
        else:
            return input_shape[:1] + [self.filters] + [
                deconv_output_length(
                    length, 
                    self.kernel_size[i], 
                    self.padding, 
                    self.strides[i])
                for i, length in enumerate(input_shape[2:])]
    
    @need_build
    def call(self, inputs, training=None):
        inputs_shape = tf.shape(inputs)
        batch_size = inputs_shape[0]
        if self.data_format == 'NCHW':
            h_axis, w_axis = 2, 3
        else:
            h_axis, w_axis = 1, 2
        height, width = None, None

        if inputs.shape.rank != None:
            dims = inputs.shape.as_list()
            height = dims[h_axis]
            width = dims[w_axis]
        height = height if height != None else inputs_shape[h_axis]
        width = width if width != None else inputs_shape[w_axis]

        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.strides

        if self.output_padding == None:
            out_pad_h = out_pad_w = None
        else:
            out_pad_h, out_pad_w = self.output_padding
        out_height = deconv_output_length(height,
                                        kernel_h,
                                        padding=self.padding,
                                        output_padding=out_pad_h,
                                        stride=stride_h)
        out_width = deconv_output_length(width,
                                        kernel_w,
                                        padding=self.padding,
                                        output_padding=out_pad_w,
                                        stride=stride_w)
        if self.data_format == 'NCHW':
            output_shape = (batch_size, self.filters, out_height, out_width)
        else:
            output_shape = (batch_size, out_height, out_width, self.filters)
        output_shape_tensor = tf.stack(output_shape)
        outputs = tf.nn.conv2d_transpose(
            inputs,
            self.kernel,
            output_shape_tensor,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format)
        if self.use_bias:
            outputs = tf.nn.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format
            )
        return outputs