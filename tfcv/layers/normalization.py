from typing import Iterable, List, Tuple, Union
import numpy as np
import tensorflow as tf

from tfcv.layers.base import Layer
from tfcv.layers.utils import need_build

__all__ = ['BatchNormalization']

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
            synchronized=False,
            trainable=True,
            name=None):
        if isinstance(beta_initializer, str):
            beta_initializer = tf.keras.initializers.get(beta_initializer)
        if isinstance(gamma_initializer, str):
            gamma_initializer = tf.keras.initializers.get(gamma_initializer)
        if isinstance(moving_mean_initializer, str):
            moving_mean_initializer = tf.keras.initializers.get(moving_mean_initializer)
        if isinstance(moving_variance_initializer, str):
            moving_variance_initializer = tf.keras.initializers.get(moving_variance_initializer)
        self._init(locals())
        super(BatchNormalization, self).__init__(name=name)

    def _assign_moving_average(self, variable, value, momentum):

        def calculate_update_delta():
            decay = tf.convert_to_tensor(
                1.0 - momentum, name='decay')
            if decay.dtype != variable.dtype.base_dtype:
                decay = tf.cast(decay, variable.dtype.base_dtype)
            update_delta = (variable - tf.cast(value, variable.dtype)) * decay
            return update_delta

        with tf.name_scope('AssignMovingAvg') as scope:
            return variable.assign_sub(calculate_update_delta(), name=scope)

    def _build(self, input_shape: Union[List, Tuple, np.ndarray]):
        if isinstance(input_shape, np.ndarray):
            input_shape = input_shape.to_list()
        else:
            input_shape = list(input_shape)
        ndims = len(input_shape)

        if isinstance(self.axis, int):
            self.axis = [self.axis]

        for idx, x in enumerate(self.axis):
            if x < 0:
                self.axis[idx] = ndims + x

        # Validate axes
        for x in self.axis:
            if x < 0 or x >= ndims:
                raise ValueError(
                    f'Invalid axis. Expected 0 <= axis < inputs.rank (with '
                    f'inputs.rank={ndims}). Received: layer.axis={self.axis}')
        if len(self.axis) != len(set(self.axis)):
            raise ValueError('Duplicate axis: %s' % (self.axis,))

        axis_to_dim = {x: input_shape[x] for x in self.axis}
        for x in axis_to_dim:
            if axis_to_dim[x] == None:
                raise ValueError('Input has undefined `axis` dimension. Received input '
                                'with shape %s. Axis value: %s' %
                                (tuple(input_shape), self.axis))
        if len(axis_to_dim) == 1:
            param_shape = (list(axis_to_dim.values())[0],)
        else:
            param_shape = [
                axis_to_dim[i] if (i in axis_to_dim) else 1 for i in range(ndims)
            ]

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
        self._output_specs = input_shape
    def compute_output_specs(self, input_shape):
        return input_shape
    @need_build
    def call(self, inputs, training=None):
        training = training and self.trainable

        inputs_dtype = inputs.dtype.base_dtype
        # inputs = tf.cast(inputs, tf.float32)TODO amp
        
        input_shape = inputs.shape
        ndims = len(input_shape)

        red_axes = [i for i in range(ndims) if i not in self.axis]
        # Broadcasting only necessary for single-axis batch norm where the axis is
        # not the last dimension
        broadcast_shape = [1] * ndims
        broadcast_shape[self.axis[0]] = input_shape.dims[self.axis[0]].value
        def _broadcast(v):
            if (v is not None and len(v.shape) != ndims and
                red_axes != list(range(ndims - 1))):
                return tf.reshape(v, broadcast_shape)
            return v
        scale, offset = _broadcast(self.gamma), _broadcast(self.beta)
        
        if training:
            mean, variance = tf.nn.moments(inputs, red_axes)
            if self.synchronized:
                import horovod.tensorflow as hvd
                assert hvd.is_initialized()
                if hvd.size()>1:
                    # Compute variance using: Var[X] = E[X^2] - E[X]^2.
                    square_of_mean = tf.math.square(mean)
                    mean_of_square = variance + square_of_mean

                    # Average stats across all workers
                    worker_stack = tf.stack([mean, mean_of_square])
                    group_stack = hvd.allreduce(worker_stack, op=hvd.Sum)
                    group_stack /= hvd.size()
                    group_mean, group_mean_of_square = tf.unstack(group_stack)
                    group_variance = group_mean_of_square - tf.math.square(group_mean)

                    mean = group_mean
                    variance = group_variance
            self._assign_moving_average(self.moving_mean, mean, self.momentum)
            self._assign_moving_average(self.moving_variance, variance, self.momentum)
            
        else:
            mean = self.moving_mean
            variance = self.moving_variance
        
        mean = tf.cast(mean, inputs.dtype)
        variance = tf.cast(variance, inputs.dtype)
        if offset is not None:
            offset = tf.cast(offset, inputs.dtype)
        if scale is not None:
            scale = tf.cast(scale, inputs.dtype)
        outputs = tf.nn.batch_normalization(inputs, _broadcast(mean),
                                        _broadcast(variance), offset, scale,
                                        self.epsilon)
        if inputs_dtype == tf.float16:
            outputs = tf.cast(outputs, inputs_dtype)
        
        return outputs