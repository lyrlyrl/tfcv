from typing import Callable, Union
import tensorflow as tf

from tfcv.layers.base import Layer
from tfcv.layers.activation import get_activation
from tfcv.layers.utils import need_build, compute_sequence_output_specs
from tfcv.layers.conv2d import Conv2D
from tfcv.layers.normalization import BatchNormalization
from tfcv.layers.linear import Linear
from tfcv.layers.pooling import MaxPooling2D, GlobalAveragePooling2D

RESNET_SPECS = {
        18: [
                ('residual', 64, 2),
                ('residual', 128, 2),
                ('residual', 256, 2),
                ('residual', 512, 2),
        ],
        34: [
                ('residual', 64, 3),
                ('residual', 128, 4),
                ('residual', 256, 6),
                ('residual', 512, 3),
        ],
        50: [
                ('bottleneck', 64, 3),
                ('bottleneck', 128, 4),
                ('bottleneck', 256, 6),
                ('bottleneck', 512, 3),
        ],
        101: [
                ('bottleneck', 64, 3),
                ('bottleneck', 128, 4),
                ('bottleneck', 256, 23),
                ('bottleneck', 512, 3),
        ],
        152: [
                ('bottleneck', 64, 3),
                ('bottleneck', 128, 8),
                ('bottleneck', 256, 36),
                ('bottleneck', 512, 3),
        ],
        200: [
                ('bottleneck', 64, 3),
                ('bottleneck', 128, 24),
                ('bottleneck', 256, 36),
                ('bottleneck', 512, 3),
        ],
        300: [
                ('bottleneck', 64, 4),
                ('bottleneck', 128, 36),
                ('bottleneck', 256, 54),
                ('bottleneck', 512, 4),
        ],
        350: [
                ('bottleneck', 64, 4),
                ('bottleneck', 128, 36),
                ('bottleneck', 256, 72),
                ('bottleneck', 512, 4),
        ],
}

class Conv2DBlock(Layer):
    default_name = 'conv2d_block'
    def __init__(self, 
                filters,
                kernel_size,
                strides,
                padding,
                kernel_initializer: Union[str, tf.keras.initializers.Initializer],
                use_bn=True,
                freeze_bn=False,
                synchronized=False,
                norm_momentum=0.9,
                norm_epsilon=1e-05,
                activation: Union[str, Callable] = None,
                trainable=True,
                name=None):
        if isinstance(activation, str):
            activation = get_activation(activation)
        self._init(locals())
        super(Conv2DBlock, self).__init__(trainable=trainable, name=name)
        self._layers['conv2d'] = Conv2D(
            filters, 
            kernel_size, 
            strides, 
            padding, 
            kernel_initializer, 
            trainable=self.trainable
        )
        if self.use_bn:
            self._layers['bn'] = BatchNormalization(
                momentum=self.norm_momentum,
                epsilon=self.norm_epsilon,
                synchronized=self.synchronized and self.trainable,
                trainable=self.trainable and (not self.freeze_bn)
            )
    def _build(self, input_shape):
        with tf.name_scope(self.name):
            self._layers['conv2d'].build(input_shape)
            output_specs = self._layers['conv2d'].output_specs
            if self.use_bn:
                self._layers['bn'].build(output_specs)
            self._output_specs = output_specs
    @need_build
    def call(self, inputs, training=None):
        outputs = self._layers['conv2d'](inputs)
        if self.use_bn:
            outputs = self._layers['bn'](outputs, training)
        if self.activation != None:
            outputs = self.activation(outputs)
        return outputs
    def compute_output_specs(self, input_shape):
        return self._layers['conv2d'].compute_output_specs(input_shape)

class BottleneckBlock(Layer):

    default_name = 'bottleneck_block'

    def __init__(
        self,
        filters,
        strides,
        use_projection=False,
        kernel_initializer='VarianceScaling',
        freeze_bn=False,
        synchronized=False,
        norm_momentum=0.9,
        norm_epsilon=1e-05,
        activation: Union[str, Callable] = 'relu',
        trainable=True, 
        name=None):
        if isinstance(activation, str):
            activation = get_activation(activation)
        self._init(locals())
        super(BottleneckBlock, self).__init__(trainable=trainable, name=name)
        self._layers['bottleneck_1'] = Conv2DBlock(
            filters, 1, 1, 'valid', kernel_initializer, True, freeze_bn, synchronized,\
                norm_momentum, norm_epsilon, activation, trainable, name='bottleneck_1'
            )
        self._layers['bottleneck_2'] = Conv2DBlock(
            filters, 3, strides, 'same', kernel_initializer, True, freeze_bn, synchronized,\
                norm_momentum, norm_epsilon, activation, trainable, name='bottleneck_2'
            )
        self._layers['bottleneck_3'] = Conv2DBlock(
            filters * 4, 1, 1, 'valid', kernel_initializer, True, freeze_bn, synchronized,\
                norm_momentum, norm_epsilon, trainable=trainable, name='bottleneck_3'
            )
        if use_projection:
            self._layers['shortcut'] = Conv2DBlock(
                filters * 4, 1, strides, 'valid', kernel_initializer, True, freeze_bn, synchronized,\
                    norm_momentum, norm_epsilon, trainable=trainable, name='shortcut'
                )
    
    def _build(self, input_shape):
        with tf.name_scope(self.name):
            if self.use_projection:
                self._layers['shortcut'].build(input_shape)
            self._layers['bottleneck_1'].build(input_shape)
            output_specs = self._layers['bottleneck_1'].output_specs
            self._layers['bottleneck_2'].build(output_specs)
            output_specs = self._layers['bottleneck_2'].output_specs
            self._layers['bottleneck_3'].build(output_specs)
            self._output_specs = self._layers['bottleneck_3'].output_specs
    
    @need_build
    def call(self, inputs, training=None):
        shortcut = inputs
        if self.use_projection:
            shortcut = self._layers['shortcut'](inputs, training=training)
        x = self._layers['bottleneck_1'](inputs, training=training)
        x = self._layers['bottleneck_2'](x, training=training)
        x = self._layers['bottleneck_3'](x, training=training)
        x = x + shortcut
        return self.activation(x)
    
    def compute_output_specs(self, input_shape):
        return compute_sequence_output_specs(
            [
                self._layers[f'bottleneck_{str(i)}'] for i in range(1, 4, 1)
            ], 
            input_shape)

class BlockGroup(Layer):

    default_name = 'blockgroup'

    def __init__(
        self, 
        filters,
        strides,
        kernel_initializer,
        freeze_bn,
        synchronized,
        norm_momentum,
        norm_epsilon,
        activation,
        block_name,
        block_repeats,
        trainable=True, 
        name=None):
        assert block_name in ['residual', 'bottleneck']
        if isinstance(kernel_initializer, str):
            kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self._init(locals())
        super(BlockGroup, self).__init__(trainable=trainable, name=name)
        if block_name == 'bottleneck':
            layer = BottleneckBlock
        for i in range(block_repeats):
            self._layers[f'block{str(i)}'] = layer(
                filters=filters,
                strides=strides if i == 0 else 1,
                use_projection=True if i == 0 else False,
                kernel_initializer=kernel_initializer,
                freeze_bn=freeze_bn,
                synchronized=synchronized,
                norm_momentum=norm_momentum,
                norm_epsilon=norm_epsilon,
                activation=activation,
                trainable=trainable,
                name=f'block{str(i)}'
            )
    def _build(self, input_shape):
        with tf.name_scope(self.name):
            for i in range(self.block_repeats):
                self._layers[f'block{str(i)}'].build(input_shape)
                input_shape = self._layers[f'block{str(i)}'].output_specs
            self._output_specs = input_shape
    @need_build
    def call(self, inputs, training=None):
        x = inputs
        for i in range(self.block_repeats):
            x = self._layers[f'block{str(i)}'](x, training=training)
        return x
    def compute_output_specs(self, input_shape):
        return compute_sequence_output_specs([self._layers[f'block{str(i)}'] for i in range(self.block_repeats)], input_shape)

class ResNet(Layer):

    default_name = 'resnet'

    def __init__(self,
        model_id,
        freeze_bn=False,
        synchronized=False,
        norm_momentum=0.9,
        norm_epsilon=1e-05,
        activation='relu',
        kernel_initializer: Union[str, tf.keras.initializers.Initializer] =
            tf.keras.initializers.VarianceScaling(
                scale=2.0, distribution='truncated_normal'),
        include_top=True,
        num_classes=1000,
        data_format='NHWC',
        trainable=True,
        name=None):
        if name == None:
            name = f'resnet{str(model_id)}'
        data_format = data_format.upper()
        assert data_format in ['NCHW', 'NHWC']
        if isinstance(kernel_initializer, str):
            kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self._init(locals())
        super(ResNet, self).__init__(trainable=trainable, name=name)

        self._layers['pre_stage'] = Conv2DBlock(
            filters=64,
            kernel_size=7,
            strides=2,
            kernel_initializer=kernel_initializer,
            padding='same',
            freeze_bn=freeze_bn,
            synchronized=synchronized,
            norm_momentum=norm_momentum,
            norm_epsilon=norm_epsilon,
            activation=activation,
            trainable=trainable,
            name='pre_stage'
        )
        self._layers['pre_maxpool'] = MaxPooling2D(
            pool_size=3, strides=2, padding='same', data_format=data_format
        )
        for i, spec in enumerate(RESNET_SPECS[model_id]):
            self._layers[f'group{str(i)}'] = BlockGroup(
                filters=spec[1],
                strides=(1 if i == 0 else 2),
                kernel_initializer=kernel_initializer,
                freeze_bn=freeze_bn,
                synchronized=synchronized,
                norm_momentum=norm_momentum,
                norm_epsilon=norm_epsilon,
                activation=activation,
                block_name=spec[0],
                block_repeats=spec[2],
                name=f'group{str(i)}'
            )
        if include_top:
            self._layers['avgpool2d'] = GlobalAveragePooling2D(data_format)
            self._layers['linear'] = Linear(num_classes, name='linear')
    
    def _build(self, input_shape):
        with tf.name_scope(self.name):
            for layer in self._layers.values():
                layer.build(input_shape)
                input_shape = layer.output_specs
            if self.include_top:
                self._output_specs = input_shape
            else:
                self._output_specs = {
                    '2': self._layers['group0'].output_specs,
                    '3': self._layers['group1'].output_specs,
                    '4': self._layers['group2'].output_specs,
                    '5': self._layers['group3'].output_specs
                }
    @need_build
    def call(self, inputs, training=None):
        x = self._layers['pre_stage'](inputs, training=training)
        x = self._layers['pre_maxpool'](x, training)

        endpoint_2 = self._layers['group0'](x, training)
        endpoint_3 = self._layers['group1'](endpoint_2, training)
        endpoint_4 = self._layers['group2'](endpoint_3, training)
        endpoint_5 = self._layers['group3'](endpoint_4, training)

        if self.include_top:
            x = self._layers['avgpool2d'](endpoint_5)
            return self._layers['linear'](x, training)
        else:
            return {
                '2': endpoint_2,
                '3': endpoint_3,
                '4': endpoint_4,
                '5': endpoint_5,}

    def compute_output_specs(self, input_shape):
        internal = self._layers['pre_stage'].compute_output_specs(input_shape)
        internal = self._layers['pre_maxpool'].compute_output_specs(internal)

        endpoint_2 = self._layers['group0'].compute_output_specs(internal)
        endpoint_3 = self._layers['group1'].compute_output_specs(endpoint_2)
        endpoint_4 = self._layers['group2'].compute_output_specs(endpoint_3)
        endpoint_5 = self._layers['group3'].compute_output_specs(endpoint_4)

        if not self.include_top:
            return {
                '2': endpoint_2,
                '3': endpoint_3,
                '4': endpoint_4,
                '5': endpoint_5
            }
        else:
            output = self._layers['avgpool2d'].compute_output_specs(endpoint_5)
            output = self._layers['linear'].compute_output_specs(output)
            return output