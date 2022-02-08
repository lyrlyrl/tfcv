import tensorflow as tf

from tfcv.classification.models.resnet import ResNet
from tfcv.detection.models.fpn import FPN

from tfcv.common import expand_image_shape

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
    def call(
        self, 
        images,
        image_info,
        training=None):
        backbone_feats = self.backbone(images, training=training)
        fpn_feats = self.fpn(backbone_feats, training=training)
        raw_scores, raw_boxes = self.head(fpn_feats)

