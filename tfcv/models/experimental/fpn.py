from typing import Mapping
import tensorflow as tf

from tfcv.layers.base import Layer
from tfcv.layers.activation import get_activation
from tfcv.layers.utils import need_build, compute_sequence_output_specs
from tfcv.layers.conv2d import Conv2D
from tfcv.layers.pooling import MaxPooling2D
from tfcv.ops import spatial_transform_ops

class FPN(Layer):

    default_name = 'fpn'

    def __init__(
        self, 
        min_level=2,
        max_level=6,
        filters=256,
        kernel_initializer='glorot_uniform',
        trainable=True, name=None):
        if isinstance(kernel_initializer, str):
            kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self._init(locals())
        super().__init__(trainable=trainable, name=name)

    def _build(self, input_shape: Mapping):
        backbone_max_level = max(list(map(int, list(input_shape.keys()))))
        self.upsample_max_level = backbone_max_level if self.max_level > backbone_max_level else self.max_level

        self._layers["stage1"] = {
            str(level): Conv2D(
                filters=self.filters,
                kernel_size=(1, 1),
                padding='same',
                kernel_initializer=self.kernel_initializer,
                name=f'l{level}',
                trainable=self.trainable
            ) for level in range(self.min_level, self.upsample_max_level + 1)
        }

        self._layers["stage2"] = {
            str(level): Conv2D(
                filters=self.filters,
                strides=(1, 1),
                kernel_size=(3, 3),
                padding='same',
                kernel_initializer=self.kernel_initializer,
                name=f'post_hoc_d{level}',
                trainable=self.trainable
            ) for level in range(self.min_level, self.upsample_max_level + 1)
        }

        if self.max_level == self.upsample_max_level + 1:
            self._layers["stage3"] = MaxPooling2D(
                pool_size=1,
                strides=2,
                padding='valid',
                name='p%d' % self.max_level
            )

        else:
            self._layers["stage3"] = {
                str(level): Conv2D(
                    filters=self.filters,
                    strides=(1, 1),
                    kernel_size=(3, 3),
                    padding='same',
                    kernel_initializer=self.kernel_initializer,
                    name=f'post_hoc_d{level}',
                    trainable=self.trainable
                ) for level in range(self.upsample_max_level + 1, self.max_level + 1)
            }
        output_specs = dict()
        with tf.name_scope(self.name):
            for level in range(self.min_level, self.upsample_max_level + 1):
                self._layers["stage1"][str(level)].build(input_shape[str(level)])
                self._layers["stage2"][str(level)].build(self._layers["stage1"][str(level)].output_specs)
                output_specs[str(level)] = self._layers["stage2"][str(level)].output_specs
            if self.max_level == self.upsample_max_level + 1:
                self._layers["stage3"].build(output_specs[str(self.max_level - 1)])
                output_specs[str(self.max_level)] = self._layers["stage3"].output_specs
            else:
                for level in range(self.upsample_max_level + 1, self.max_level + 1):
                    self._layers["stage3"][str(level)].build(output_specs[str(level-1)])
                    output_specs[str(level)] = self._layers["stage3"][str(level)].output_specs
            self._output_specs = output_specs

    @need_build
    def compute_output_specs(self, input_shape):
        output_specs = {
            str(level): compute_sequence_output_specs(
                [self._layers["stage1"][str(level)], self._layers["stage2"][str(level)]], input_shape[str(level)]
            )
            for level in range(self.min_level, self.upsample_max_level + 1)
        }
        if self.max_level == self.upsample_max_level + 1:
            output_specs[str(self.max_level)] = self._layers["stage3"]
        
        for level in range(self.min_level, self.upsample_max_level + 1):
            self._layers["stage1"][str(level)].build(input_shape[str(level)])
            self._layers["stage2"][str(level)].build(self._layers["stage1"][str(level)].output_specs)
            output_specs[str(level)] = self._layers["stage2"][str(level)].output_specs
        if self.max_level == self.upsample_max_level + 1:
            self._layers["stage3"].build(output_specs[str(self.max_level - 1)])
            output_specs[str(self.max_level)] = self._layers["stage3"].output_specs
        else:
            for level in range(self.upsample_max_level + 1, self.max_level + 1):
                self._layers["stage3"][str(level)].build(output_specs[str(level-1)])
                output_specs[str(level)] = self._layers["stage3"][str(level)].output_specs
                
    @need_build
    def call(self, inputs, training=None):
        feats_lateral = {
            str(level): self._layers['stage1'][str(level)](
                inputs[str(level)], training=training
            ) for level in range(self.min_level, self.upsample_max_level + 1)
        }
        # add top-down path
        feats = {str(self.upsample_max_level): feats_lateral[str(self.upsample_max_level)]}
        for level in range(self.upsample_max_level - 1, self.min_level - 1, -1):
            feats[str(level)] = spatial_transform_ops.nearest_upsampling(
                feats[str(level + 1)], 2
            ) + feats_lateral[str(level)]
        
        # add post-hoc 3x3 convolution kernel
        for level in range(self.min_level, self.upsample_max_level + 1):
            feats[str(level)] = self._layers["stage2"][str(level)](feats[str(level)])

        if self.max_level == self.upsample_max_level + 1:
            feats[str(self.max_level)] = self._layers["stage3"](feats[str(self.max_level - 1)])

        else:
            for level in range(self.upsample_max_level + 1, self.max_level + 1):
                feats[str(level)] = self._layers["stage3"][str(level)](feats[str(level - 1)])

        return feats
