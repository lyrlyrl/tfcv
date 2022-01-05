from typing import Mapping
import tensorflow as tf

from tfcv.layers.base import Layer
from tfcv.layers.activation import get_activation
from tfcv.layers.utils import need_build
from tfcv.layers.conv2d import Conv2D
from tfcv.layers.pooling import MaxPooling2D
from tfcv.models.experimental.heads import RPNHead, FCBoxHead, MultilevelRPNHead
from tfcv.models.experimental.fpn import FPN
from tfcv.models.experimental.resnet import ResNet
from tfcv.ops import spatial_transform_ops

class GenelizedRCNN(Layer):

    default_name = 'genelized_rcnn'

    def __init__(
        self, 
        cfg, 
        trainable=True,
        name=None):
        self._init(locals())
        super(GenelizedRCNN, self).__init__(trainable=trainable, name=name)

        self.backbone = ResNet(
            self.cfg.backbone.resnet_id,
            include_top=False)
        self.fpn = FPN(
            min_level=self.cfg.min_level,
            max_level=self.cfg.max_level)
        self.rpn_head = MultilevelRPNHead(
            min_level=self.cfg.min_level,
            max_level=self.cfg.max_level,
            num_anchors=len(self.cfg.anchor.aspect_ratios * self.cfg.anchor.num_scales),
            num_filters=256,
            name='rpn_head')
        self.box_head = FCBoxHead(
            num_classes=self.cfg.num_classes,
            mlp_head_dim=self.cfg.frcnn.mlp_head_dim,
            name='box_head'
        )
    @need_build
    def call(self, images, training=None):
        backbone_feats = self.backbone(images, training=training)
        fpn_feats = self.fpn(backbone_feats, training=training)
        rpn_score_outputs, rpn_box_outputs = self.rpn_head(
            fpn_feats, training=training
        )
        return (rpn_score_outputs, rpn_box_outputs)

    def _build(self, input_shape):
        batch_size, image_height, image_width, _ = input_shape
        with tf.name_scope(self.name):
            self.backbone.build(input_shape)
            self.fpn.build(self.backbone.output_specs)
            self.rpn_head.build(self.fpn.output_specs)
            self._output_specs = self.rpn_head.output_specs
    
    def compute_output_specs(self, input_shape):
        pass
            
