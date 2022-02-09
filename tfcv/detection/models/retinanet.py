import tensorflow as tf
import numpy as np

from tfcv.layers.normalization import BatchNormalization

from tfcv.classification.models.resnet import ResNet
from tfcv.detection.models.fpn import FPN

from tfcv.common import expand_image_shape

class RetinanetHead(tf.keras.layers.Layer):
    def __init__(
        self, 
        min_level: int,
        max_level: int,
        num_classes: int,
        num_anchors: int,
        num_convs: int = 4,
        num_filters: int = 256,
        norm_momentum: float = 0.99,
        norm_epsilon: float = 0.001,
        **kwargs):
        super().__init__(**kwargs)
        self._min_level = min_level
        self._max_level = max_level

        # Class net.
        self._cls_convs = []
        self._cls_norms = []

        for level in range(min_level, max_level + 1):
            this_level_cls_norms = []
            for i in range(num_convs):
                if level == min_level:
                    self._cls_convs.append(
                        tf.keras.layers.Conv2D(
                            name='classnet-conv_{}'.format(i), 
                            filters=num_filters,
                            kernel_size=3,
                            padding='same',
                            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
                            bias_initializer=tf.zeros_initializer())
                    )
                this_level_cls_norms.append(
                    BatchNormalization(
                        name='classnet-conv-norm_{}_{}'.format(level, i), 
                        axis=-1 if tf.keras.backend.image_data_format() == 'channels_last' else 1,
                        momentum=norm_momentum,
                        epsilon=norm_epsilon,
                        trainable=kwargs.get('trainable', True))
                    )
        
        self._classifier = tf.keras.layers.Conv2D(
            name='scores', 
            filters=num_classes*num_anchors,
            kernel_size=3,
            padding='same',
            bias_initializer=tf.constant_initializer(-np.log((1 - 0.01) / 0.01)),
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=1e-5))
        
        # Box net.
        self._box_convs = []
        self._box_norms = []

        for level in range(min_level, max_level + 1):
            this_level_box_norms = []
            for i in range(num_convs):
                if level == min_level:
                    self._box_convs.append(
                        tf.keras.layers.Conv2D(
                            name='boxnet-conv_{}'.format(i), 
                            filters=num_filters,
                            kernel_size=3,
                            padding='same',
                            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
                            bias_initializer=tf.zeros_initializer()
                        )
                    )
                this_level_box_norms.append(
                    BatchNormalization(
                        name='boxnet-conv-norm_{}_{}'.format(level, i),
                        axis=-1 if tf.keras.backend.image_data_format() == 'channels_last' else 1,
                        momentum=norm_momentum,
                        epsilon=norm_epsilon,
                        trainable=kwargs.get('trainable', True))
                    )
            self._box_norms.append(this_level_box_norms)
            
        self._box_regressor = tf.keras.layers.Conv2D(
            name='boxes', 
            filters=num_anchors*4,
            kernel_size=3,
            padding='same',
            bias_initializer=tf.zeros_initializer(),
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=1e-5))

    def call(self, features, training=None):

        scores = {}
        boxes = {}

        for i, level in enumerate(
            range(self._min_level, self._max_level + 1)):
            this_level_features = features[str(level)]

            # class net.
            x = this_level_features
            for conv, norm in zip(self._cls_convs, self._cls_norms[i]):
                x = conv(x)
                x = norm(x)
                x = tf.nn.relu(x)
            scores[str(level)] = self._classifier(x)

            # box net.
            x = this_level_features
            for conv, norm in zip(self._box_convs, self._box_norms[i]):
                x = conv(x)
                x = norm(x)
                x = tf.nn.relu(x)
            boxes[str(level)] = self._box_regressor(x)
        
        return scores, boxes


class Retinanet(tf.keras.Model):

    def __init__(self, config, name='retinanet', *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)

        self.cfg = config

        self.backbone = ResNet(
            self.cfg.backbone.resnet_id,
            input_shape=expand_image_shape(self.cfg.data.image_size),
            freeze_at=0 if self.cfg.from_scratch else 2,
            freeze_bn=False if self.cfg.from_scratch else True,
            include_top=False,
            pretrained=False if self.cfg.from_scratch else 'imagenet')

        self.fpn = FPN(
            self.backbone.output,
            min_level=self.cfg.min_level,
            max_level=self.cfg.max_level
        )

        self.head = RetinanetHead(
            min_level=self.cfg.min_level,
            max_level=self.cfg.max_level,
            num_classes=self.cfg.num_classes,
            num_anchors=len(self.cfg.anchor.aspect_ratios * self.cfg.anchor.num_scales),
        )
        
    def call(
        self, 
        images,
        training=None):

        backbone_feats = self.backbone(images, training=training)

        fpn_feats = self.fpn(backbone_feats, training=training)

        raw_scores, raw_boxes = self.head(fpn_feats)

        return {'cls_outputs': raw_scores, 'box_outputs': raw_boxes}

