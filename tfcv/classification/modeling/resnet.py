
"""Contains definitions of Residual Networks.

Residual networks (ResNets) were proposed in:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
        Deep Residual Learning for Image Recognition. arXiv:1512.03385
"""

import tensorflow as tf

import tfcv
from tfcv.classification.modeling import nn_blocks

layers = tf.keras.layers

__all__ = ['ResNet']

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

RESNET_PRETRAINED = {
    50: {
        'imagenet': 'resnet50_imagenet'
    }
}

class ResNet(tfcv.Model):
    """Class to build ResNet family model."""

    def __init__(self,
        model_id,
        input_shape=[None, None, 3],
        freeze_at=-1,
        freeze_bn=False,
        use_sync_bn=False,
        norm_momentum=0.9,
        norm_epsilon=1e-05,
        activation='relu',
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            scale=2.0, distribution='truncated_normal'),
        include_top=True,
        num_classes=1001,
        pretrained: str = 'imagenet',
        **kwargs):
        """ResNet initialization function.

        Args:
            model_id: `int` depth of ResNet backbone model.
            input_specs: `tf.keras.layers.InputSpec` specs of the input tensor.
            activation: `str` name of the activation function.
            use_sync_bn: if True, use synchronized batch normalization.
            norm_momentum: `float` normalization omentum for the moving average.
            norm_epsilon: `float` small float added to variance to avoid dividing by
                zero.
            kernel_initializer: kernel_initializer for convolutional layers.
            **kwargs: keyword arguments to be passed.
        """
        self._model_id = model_id
        self._kernel_initializer = kernel_initializer
        if freeze_bn:
            use_sync_bn = False
        # self._activation = get_activation(use_keras_layer=True)

            # Build ResNet.
        if len(input_shape) == 2:
            input_shape = input_shape+[3]
        inputs = tf.keras.Input(shape=input_shape)

        x = nn_blocks.Conv2DBlock(
            filters=64,
            kernel_size=7,
            strides=2,
            kernel_initializer=kernel_initializer,
            padding='same',
            freeze_bn=freeze_bn,
            use_sync_bn=use_sync_bn,
            norm_momentum=norm_momentum,
            norm_epsilon=norm_epsilon,
            activation=activation,
            trainable=(freeze_at-2 < 0),
            name='resnet{}/pre_stage'.format(model_id)
        )(inputs)

        x = layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)

        endpoints = {}
        for i, spec in enumerate(RESNET_SPECS[model_id]):
            x = nn_blocks.block_group(
                    x,
                    filters=spec[1],
                    strides=(1 if i == 0 else 2),
                    kernel_initializer=kernel_initializer,
                    freeze_bn=freeze_bn,
                    use_sync_bn=use_sync_bn,
                    norm_momentum=norm_momentum,
                    norm_epsilon=norm_epsilon,
                    activation=activation,
                    block_name=spec[0],
                    block_repeats=spec[2],
                    trainable=(freeze_at-2 < i),
                    name=f'resnet{model_id}/group{i}')
            endpoints[str(i + 2)] = x

        if include_top:
            raise
            super(ResNet, self).__init__(inputs=inputs, outputs=endpoints, name=f'resnet{model_id}', **kwargs)
            if pretrained:
                assert num_classes==1001, 'imagenet pretrained classification model must be 1001 classes'
                self.load(RESNET_PRETRAINED[model_id][pretrained])
        else:
            super(ResNet, self).__init__(inputs=inputs, outputs=endpoints, name=f'resnet{model_id}', **kwargs)
            if pretrained:
                self.load(RESNET_PRETRAINED[model_id][pretrained])
    def get_config(self):
        config_dict = {
                'model_id': self._model_id,
                'kernel_initializer': self._kernel_initializer,
        }
        return config_dict

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**config)

    @staticmethod
    def freeze_at(stage, model_id):
        return f'resnet{model_id}/(conv2d|batchnorm|btlnck_block_0.*)'