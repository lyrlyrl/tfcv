import tensorflow as tf

from tfcv.layers.normalization import BatchNormalization

class Conv2DBlock(tf.keras.layers.Layer):
    def __init__(self,
                filters,
                strides,
                kernel_size,
                padding,
                kernel_initializer,
                use_bn=True,
                freeze_bn=False,
                norm_momentum=0.9,
                norm_epsilon=1e-05,
                activation=None,
                # trainable=True,
                name=None,
                **kwargs):
        if name == None:
            kwargs['name'] = 'conv2d_block'
        else:
            kwargs['name'] = name
        super(Conv2DBlock, self).__init__(**kwargs)
        self._filters = filters
        self._strides = strides
        self._kernel_size = kernel_size
        self._padding = padding
        self._kernel_initializer = kernel_initializer
        
        self._conv = tf.keras.layers.Conv2D(
            filters=self._filters,
            kernel_size=self._kernel_size,
            strides=self._strides,
            padding=self._padding,
            use_bias=not use_bn,
            kernel_initializer=self._kernel_initializer,
            # trainable=trainable,
            name='conv2d')

        if use_bn:
            if tf.keras.backend.image_data_format() == 'channels_last':
                bn_axis = -1
            else:
                bn_axis = 1
            self._bn = BatchNormalization(
                axis = bn_axis,
                momentum = norm_momentum,
                epsilon = norm_epsilon,
                trainable = not freeze_bn,
                name='bn')
        else:
            self._bn = None
        
        if activation:
            self._activation = tf.keras.layers.Activation(activation)
        else:
            self._activation = None

    def call(self, inputs, training=None):
        nets = self._conv(inputs)
        if self._bn:
            nets = self._bn(nets, training=training)
        if self._activation:
            nets = self._activation(nets)
        return nets