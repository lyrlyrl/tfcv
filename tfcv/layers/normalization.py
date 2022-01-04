from typing import Iterable, List, Tuple, Union

import tensorflow as tf
from tensorflow.python.ops.variables import Variable

from tfcv.layers.base import Layer
from tfcv.layers.utils import need_build

class BatchNormalization(Layer):
    default_name = 'bn'

    def __init__(
            self, 
            axis: int = -1,
            momentum=0.99, 
            epsilon=0.001, 
            center=True, 
            scale=True,
            beta_initializer: Union[str, tf.keras.initializers.Initializer] ='zeros', 
            gamma_initializer: Union[str, tf.keras.initializers.Initializer] ='ones',
            moving_mean_initializer: Union[str, tf.keras.initializers.Initializer] ='zeros',
            moving_variance_initializer: Union[str, tf.keras.initializers.Initializer] ='ones',
            sync_statistics=None,
            name=None):
        if isinstance(beta_initializer, str):
            beta_initializer = tf.keras.initializers.get(beta_initializer)
        if isinstance(gamma_initializer, str):
            gamma_initializer = tf.keras.initializers.get(gamma_initializer)
        if isinstance(moving_mean_initializer, str):
            moving_mean_initializer = tf.keras.initializers.get(moving_mean_initializer)
        if isinstance(beta_initializer, str):
            moving_variance_initializer = tf.keras.initializers.get(moving_variance_initializer)
        self._init(locals())
        self.fused = True
        super(BatchNormalization, self).__init__(name=name)
    
    def _build(self, input_shape):
        ndims = len(input_shape)
        assert ndims in [2, 4]
        
        if self.axis < 0:
            self.axis = ndims + self.axis
        assert self.axis in [1, 3]

        if self.axis > ndims:
            raise ValueError(
                    f'Invalid axis. Expected 0 <= axis < inputs.rank (with '
                    f'inputs.rank={ndims}). Received: layer.axis={self.axis}')
        
        if (self.axis == 1 or self.axis == 3) and ndims == 4:
            self.fused=True
        else:
            self.fused=False

        param_shape = (input_shape[self.axis],)

        with tf.name_scope(self.name):
            if self.scale:
                self.gamma = tf.Variable(
                    initial_value=self.gamma_initializer(
                        shape = param_shape,
                        dtype = tf.float32
                    ),
                    trainable=True,
                    name='gamma'
                )
            else:
                self.gamma = None
                if self.fused:
                    self._gamma_const = tf.constant(
                        1.0, dtype=tf.float32, shape=param_shape)

            if self.center:
                self.beta = tf.Variable(
                    initial_value = self.beta_initializer(
                        shape = param_shape,
                        dtype = tf.float32
                    ),
                    trainable=True,
                    name='beta'
                )
            else:
                self.beta = None
                if self.fused:
                    self._beta_const = tf.constant(
                        0.0, dtype=tf.float32, shape=param_shape)
            
            self.moving_mean = tf.Variable(
                initial_value = self.moving_mean_initializer(
                    shape = param_shape,
                    dtype = tf.float32
                ),
                trainable=False,
                name='moving_mean'
            )

            self.moving_variance = tf.Variable(
                initial_value = self.moving_variance_initializer(
                    shape = param_shape,
                    dtype = tf.float32
                ),
                trainable=False,
                name='moving_variance'
            )

    def _fused_batch_norm(self, inputs, training):
        beta = self.beta if self.center else self._beta_const
        gamma = self.gamma if self.scale else self._gamma_const

        exponential_avg_factor = 1.0 - self.momentum

        def _fused_batch_norm_training():
            return tf.compat.v1.nn.fused_batch_norm(
                inputs,
                gamma,
                beta,
                mean=self.moving_mean,
                variance=self.moving_variance,
                epsilon=self.epsilon,
                is_training=True,
                data_format=self._data_format,
                exponential_avg_factor=exponential_avg_factor)
        def _fused_batch_norm_training_empty():
            return inputs, self.moving_mean, self.moving_variance
        if training:
            input_batch_size = tf.shape(inputs)[0]
            output, mean, variance = tf.cond(
                input_batch_size>0,
                _fused_batch_norm_training,
                _fused_batch_norm_training_empty
            )
        else:
            output, mean, variance = tf.compat.v1.nn.fused_batch_norm(
                inputs,
                gamma,
                beta,
                mean=self.moving_mean,
                variance=self.moving_variance,
                epsilon=self.epsilon,
                is_training=False,
                data_format=self._data_format,
            )
        if training:
            self.moving_mean.assign(mean)
            self.moving_variance.assign(variance)
        return output
    
    @need_build
    def call(self, inputs, training=None):
        if self.fused and training and (self.sync_statistics == None):
            outputs = self._fused_batch_norm(inputs, training=training)
            return outputs
        else:
            input_shape = inputs.shape
            ndims = len(input_shape)

            red_axis = [0] if ndims == 2 else ([0, 2, 3] if self.axis == 1 else [0, 1, 2])