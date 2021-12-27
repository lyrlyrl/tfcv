# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Feature Pyramid Network.

Feature Pyramid Networks were proposed in:
[1] Tsung-Yi Lin, Piotr Dollar, Ross Girshick, Kaiming He, Bharath Hariharan,
    , and Serge Belongie
    Feature Pyramid Networks for Object Detection. CVPR 2017.
"""

from __future__ import absolute_import, division, print_function

import tensorflow as tf

from tfcv.ops import spatial_transform_ops

class FPNNetwork(tf.keras.models.Model):
    def __init__(self, min_level=3, max_level=7, filters=256, trainable=True):
        """Generates multiple scale feature pyramid (FPN).

        Args:
        feats_bottom_up: a dictionary of tensor with level as keys and bottom up
          feature tensors as values. They are the features to generate FPN features.
        min_level: the minimum level number to generate FPN features.
        max_level: the maximum level number to generate FPN features.
        filters: the FPN filter size.

        Returns:
        feats: a dictionary of tensor with level as keys and the generated FPN
          features as values.
        """
        super().__init__(name="fpn", trainable=trainable)

        self._local_layers = dict()

        self._min_level = min_level
        self._max_level = max_level

        self._filters = filters

        self._backbone_max_level = 5  # max(feats_bottom_up.keys())
        self._upsample_max_level = (
            self._backbone_max_level if self._max_level > self._backbone_max_level else self._max_level
        )

        self._local_layers["stage1"] = dict()
        for level in range(self._min_level, self._upsample_max_level + 1):
            self._local_layers["stage1"][str(level)] = tf.keras.layers.Conv2D(
                filters=self._filters,
                kernel_size=(1, 1),
                padding='same',
                name=f'l{level}',
                trainable=trainable
            )

        self._local_layers["stage2"] = dict()
        # add post-hoc 3x3 convolution kernel
        for level in range(self._min_level, self._upsample_max_level + 1):
            self._local_layers["stage2"][str(level)] = tf.keras.layers.Conv2D(
                filters=self._filters,
                strides=(1, 1),
                kernel_size=(3, 3),
                padding='same',
                name=f'post_hoc_d{level}',
                trainable=trainable
            )

        self._local_layers["stage3_1"] = dict()
        self._local_layers["stage3_2"] = dict()

        if self._max_level == self._upsample_max_level + 1:
            self._local_layers["stage3_1"] = tf.keras.layers.MaxPool2D(
                pool_size=1,
                strides=2,
                padding='valid',
                name='p%d' % self._max_level,
                trainable=trainable
            )

        else:
            for level in range(self._upsample_max_level + 1, self._max_level + 1):
                self._local_layers["stage3_2"][str(level)] = tf.keras.layers.Conv2D(
                    filters=self._filters,
                    strides=(2, 2),
                    kernel_size=(3, 3),
                    padding='same',
                    name=f'p{level}',
                    trainable=trainable
                )

    def call(self, inputs, *args, **kwargs):

        feats_bottom_up = inputs

        # lateral connections
        feats_lateral = {}

        for level in range(self._min_level, self._upsample_max_level + 1):
            feats_lateral[level] = self._local_layers["stage1"][str(level)](feats_bottom_up[level])

        # add top-down path
        feats = {self._upsample_max_level: feats_lateral[self._upsample_max_level]}

        for level in range(self._upsample_max_level - 1, self._min_level - 1, -1):
            feats[level] = spatial_transform_ops.nearest_upsampling(
                feats[level + 1], 2
            ) + feats_lateral[level]

        # add post-hoc 3x3 convolution kernel
        for level in range(self._min_level, self._upsample_max_level + 1):
            feats[level] = self._local_layers["stage2"][str(level)](feats[level])

        if self._max_level == self._upsample_max_level + 1:
            feats[self._max_level] = self._local_layers["stage3_1"](feats[self._max_level - 1])

        else:
            for level in range(self._upsample_max_level + 1, self._max_level + 1):
                feats[level] = self._local_layers["stage3_2"][str(level)](feats[level - 1])

        return feats

class FPN(tf.keras.Model):
    """Feature pyramid network."""

    def __init__(self,
                input_specs,
                min_level=3,
                max_level=7,
                num_filters=256,
                kernel_initializer='glorot_uniform',
                activation='relu',
                **kwargs):
        """FPN initialization function.

        Args:
            input_specs: `dict` input specifications. A dictionary consists of
                {level: TensorShape} from a backbone.
            min_level: `int` minimum level in FPN output feature maps.
            max_level: `int` maximum level in FPN output feature maps.
            num_filters: `int` number of filters in FPN layers.
            activation: `str` name of the activation function.
            kernel_initializer: kernel_initializer for convolutional layers.
            kernel_regularizer: tf.keras.regularizers.Regularizer object for Conv2D.
            bias_regularizer: tf.keras.regularizers.Regularizer object for Conv2d.
            **kwargs: keyword arguments to be passed.
        """
        self._config_dict = {
                'input_specs': input_specs,
                'min_level': min_level,
                'max_level': max_level,
                'num_filters': num_filters,
                'kernel_initializer': kernel_initializer,
        }

        conv2d = tf.keras.layers.Conv2D

        # Get input feature pyramid from backbone.
        inputs = self._build_input_pyramid(input_specs, min_level)
        upsample_max_level = min(max(list(map(int, inputs.keys()))), max_level)

        # Build lateral connections.
        feats_lateral = {}
        for level in range(min_level, upsample_max_level + 1):
            feats_lateral[str(level)] = conv2d(
                filters=num_filters,
                kernel_size=1,
                padding='same',
                kernel_initializer=kernel_initializer,
                name='fpn/lateral_1x1_c{}'.format(level))(
                        inputs[str(level)])

        # Build top-down path.
        feats = {str(upsample_max_level): feats_lateral[str(upsample_max_level)]}
        for level in range(upsample_max_level - 1, min_level - 1, -1):
            feats[str(level)] = spatial_transform_ops.nearest_upsampling(
                feats[str(level + 1)], 2) + feats_lateral[str(level)]

        for level in range(min_level, upsample_max_level + 1):
            feats[str(level)] = conv2d(
                filters=num_filters,
                strides=1,
                kernel_size=3,
                padding='same',
                kernel_initializer=kernel_initializer,
                name='fpn/posthoc_3x3_p{}'.format(level))(feats[str(level)])


        if max_level == upsample_max_level + 1:
            feats[str(max_level)] = tf.keras.layers.MaxPool2D(
                pool_size=1,
                strides=2,
                padding='valid',
                name='fpn/coarser_3x3_p%d' % max_level,
            )(feats[str(max_level - 1)])
        else:
            for level in range(upsample_max_level + 1, max_level + 1):
                feats_in = feats[str(level - 1)]
                if level > upsample_max_level + 1:
                    feats_in = tf.keras.layers.Activation(activation)(feats_in)
                feats[str(level)] = conv2d(
                    filters=num_filters,
                    strides=2,
                    kernel_size=3,
                    padding='same',
                    kernel_initializer=kernel_initializer,
                    name='fpn/coarser_3x3_p{}'.format(level))(feats_in)

        super(FPN, self).__init__(inputs=inputs, outputs=feats, **kwargs)

    def _build_input_pyramid(self, input_specs, min_level):
        assert isinstance(input_specs, dict)
        if min(list(map(int, input_specs.keys()))) > min_level:
            raise ValueError(
                    'Backbone min level should be less or equal to FPN min level')

        inputs = {}
        for level, spec in input_specs.items():
            inputs[level] = tf.keras.Input(shape=spec.shape[1:])
        return inputs

    def get_config(self):
        return self._config_dict

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**config)