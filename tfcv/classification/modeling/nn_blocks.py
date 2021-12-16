
"""Contains common building blocks for neural networks."""
import tensorflow as tf

from tfcv.layers import SyncBatchNormalization

class Conv2DBlock(tf.keras.layers.Layer):
    def __init__(self,
                filters,
                strides,
                kernel_size,
                padding,
                kernel_initializer,
                use_bn=True,
                freeze_bn=False,
                use_sync_bn=False,
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
            layer = SyncBatchNormalization if use_sync_bn else tf.keras.layers.BatchNormalization
            self._bn = layer(
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

class ResidualBlock(tf.keras.layers.Layer):
    """A residual block."""

    def __init__(self,
            filters,
            strides,
            use_projection=False,
            kernel_initializer='VarianceScaling',
            use_sync_bn=False,
            norm_momentum=0.9,
            norm_epsilon=1e-05,
            activation='relu',
            **kwargs):
        """A residual block with BN after convolutions.

        Args:
            filters: `int` number of filters for the first two convolutions. Note that
                the third and final convolution will use 4 times as many filters.
            strides: `int` block stride. If greater than 1, this block will ultimately
                downsample the input.
            use_projection: `bool` for whether this block should use a projection
                shortcut (versus the default identity shortcut). This is usually `True`
                for the first block of a block group, which may change the number of
                filters and the resolution.
            se_ratio: `float` or None. Ratio of the Squeeze-and-Excitation layer.
            stochastic_depth_drop_rate: `float` or None. if not None, drop rate for
                the stochastic depth layer.
            kernel_initializer: kernel_initializer for convolutional layers.
            kernel_regularizer: tf.keras.regularizers.Regularizer object for Conv2D.
                                                    Default to None.
            bias_regularizer: tf.keras.regularizers.Regularizer object for Conv2d.
                                                Default to None.
            activation: `str` name of the activation function.
            **kwargs: keyword arguments to be passed.
        """
        super(ResidualBlock, self).__init__(**kwargs)

        self._filters = filters
        self._strides = strides
        self._use_projection = use_projection
        self._activation = activation
        self._kernel_initializer = kernel_initializer
        self._activation_fn = tf.keras.layers.Activation(activation)

    def build(self, input_shape):
        if self._use_projection:
            self._shortcut = Conv2DBlock(
                filters=self._filters,
                kernel_size=1,
                padding='valid',
                strides=self._strides,
                use_bias=False,
                kernel_initializer=self._kernel_initializer,
                kernel_regularizer=self._kernel_regularizer,
                bias_regularizer=self._bias_regularizer,
                name='shortcut'
            )
        self._r1 = Conv2DBlock(
                filters=self._filters,
                kernel_size=3,
                padding='same',
                strides=self._strides,
                use_bias=False,
                kernel_initializer=self._kernel_initializer,
                kernel_regularizer=self._kernel_regularizer,
                bias_regularizer=self._bias_regularizer,
                activation_fn=self._activation_fn,
                name='residual_1'
            )

        self._r2 = Conv2DBlock(
                filters=self._filters,
                kernel_size=3,
                padding='same',
                strides=1,
                use_bias=False,
                kernel_initializer=self._kernel_initializer,
                kernel_regularizer=self._kernel_regularizer,
                bias_regularizer=self._bias_regularizer,
                name='residual_2'
            )


        super(ResidualBlock, self).build(input_shape)

    def call(self, inputs, training=None):
        shortcut = inputs
        if self._use_projection:
            shortcut = self._shortcut(shortcut)

        x = self._r1(inputs, training=training)

        x = self._r2(x, training=training)

        return self._activation_fn(x + shortcut)

class BottleneckBlock(tf.keras.layers.Layer):
    """A standard bottleneck block."""

    def __init__(self,
                filters,
                strides,
                use_projection=False,
                kernel_initializer='VarianceScaling',
                freeze_bn=False,
                use_sync_bn=False,
                norm_momentum=0.9,
                norm_epsilon=1e-05,
                activation='relu',
                **kwargs):
        """A standard bottleneck block with BN after convolutions.

        Args:
            filters: `int` number of filters for the first two convolutions. Note that
                the third and final convolution will use 4 times as many filters.
            strides: `int` block stride. If greater than 1, this block will ultimately
                downsample the input.
            dilation_rate: `int` dilation_rate of convolutions. Default to 1.
            use_projection: `bool` for whether this block should use a projection
                shortcut (versus the default identity shortcut). This is usually `True`
                for the first block of a block group, which may change the number of
                filters and the resolution.
            se_ratio: `float` or None. Ratio of the Squeeze-and-Excitation layer.
            stochastic_depth_drop_rate: `float` or None. if not None, drop rate for
                the stochastic depth layer.
            kernel_initializer: kernel_initializer for convolutional layers.
            kernel_regularizer: tf.keras.regularizers.Regularizer object for Conv2D.
                                                    Default to None.
            bias_regularizer: tf.keras.regularizers.Regularizer object for Conv2d.
                                                Default to None.
            activation: `str` name of the activation function.
            **kwargs: keyword arguments to be passed.
        """
        super(BottleneckBlock, self).__init__(**kwargs)

        self._filters = filters
        self._strides = strides
        self._use_projection = use_projection
        self._activation = activation
        self._kernel_initializer = kernel_initializer

        if self._use_projection:
            self._shortcut = Conv2DBlock(
                filters=self._filters * 4,
                kernel_size=1,
                strides=self._strides,
                padding='valid',
                kernel_initializer=self._kernel_initializer,
                freeze_bn=freeze_bn,
                use_sync_bn=use_sync_bn,
                norm_momentum=norm_momentum,
                norm_epsilon=norm_epsilon,
                name='shortcut'
            )
        self._b1 = Conv2DBlock(
            filters=self._filters,
            kernel_size=1,
            strides=1,
            padding='valid',
            kernel_initializer=self._kernel_initializer,
            freeze_bn=freeze_bn,
            use_sync_bn=use_sync_bn,
            norm_momentum=norm_momentum,
            norm_epsilon=norm_epsilon,
            activation=activation,
            name='bottleneck_1'
        )

        self._b2 = Conv2DBlock(
            filters=self._filters,
            kernel_size=3,
            strides=self._strides,
            padding='same',
            kernel_initializer=self._kernel_initializer,
            freeze_bn=freeze_bn,
            use_sync_bn=use_sync_bn,
            norm_momentum=norm_momentum,
            norm_epsilon=norm_epsilon,
            activation=activation,
            name='bottleneck_2'
        )

        self._b3 = Conv2DBlock(
            filters=self._filters * 4,
            kernel_size=1,
            strides=1,
            padding='valid',
            kernel_initializer=self._kernel_initializer,
            freeze_bn=freeze_bn,
            use_sync_bn=use_sync_bn,
            norm_momentum=norm_momentum,
            norm_epsilon=norm_epsilon,
            name='bottleneck_3'
        )

        self._add = tf.keras.layers.Add()
        self._activation = tf.keras.layers.Activation(activation)

    def call(self, inputs, training=None):
        shortcut = inputs
        if self._use_projection:
            shortcut = self._shortcut(shortcut, training=training)

        x = self._b1(inputs, training=training)

        x = self._b2(x, training=training)

        x = self._b3(x, training=training)

        x = self._add([x, shortcut])

        return self._activation(x)

def block_group(
        x,
        filters,
        strides,
        kernel_initializer,
        freeze_bn,
        use_sync_bn,
        norm_momentum,
        norm_epsilon,
        activation,
        block_name,
        block_repeats,
        activate_after=False,
        trainable=True,
        name='block_group'
    ):
    if block_name == 'residual':
        block_fn = ResidualBlock
    elif block_name == 'bottleneck':
        block_fn = BottleneckBlock
    else:
        raise ValueError('Block fn `{}` is not supported.'.format(block_name))  

    x = block_fn(
        filters=filters,
        strides=strides,
        use_projection=True,
        kernel_initializer=kernel_initializer,
        freeze_bn=freeze_bn,
        use_sync_bn=use_sync_bn,
        norm_momentum=norm_momentum,
        norm_epsilon=norm_epsilon,
        activation=activation,
        trainable=trainable,
        name=f'{name}/block0'
        )(x)

    for i in range(1, block_repeats):
        x = block_fn(
                filters=filters,
                strides=1,
                use_projection=False,
                kernel_initializer=kernel_initializer,
                freeze_bn=freeze_bn,
                use_sync_bn=use_sync_bn,
                norm_momentum=norm_momentum,
                norm_epsilon=norm_epsilon,
                activation=activation,
                trainable=trainable,
                name=f'{name}/block{str(i)}')(x)
    if activate_after:
        return tf.keras.layers.Activation('linear')(x)
    else:
        return x