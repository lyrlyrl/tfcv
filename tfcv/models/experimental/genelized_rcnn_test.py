import tensorflow as tf
from tfcv.config import AttrDict
from tfcv.models.experimental.genelized_rcnn import GenelizedRCNN
if __name__ == '__main__':
    config = AttrDict()
    config.backbone.resnet_id=50
    config.min_level = 2
    config.max_level = 6
    config.anchor.aspect_ratios=[1,2,3]
    config.anchor.num_scales=1
    config.num_classes=91
    config.frcnn.mlp_head_dim=1024
    model = GenelizedRCNN(config)
    model.build([None, 832, 1344, 3])
    # print(model(tf.ones([3, 832, 1344, 3]), training=False))