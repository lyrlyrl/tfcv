from tfcv.layers.base import Layer

from tfcv.ops import spatial_transform_ops

class RoiAligner(Layer):
    def __init__(self, 
        output_size,
        name=None):
        self._init(locals())
        super().__init__(trainable=False, name=name)
    def call(self, features, boxes, training=None):
        return spatial_transform_ops.multilevel_crop_and_resize(
            features=features,
            boxes=boxes,
            output_size=self.output_size,
            training=training
        )
    def _build(self, inputs, training=None):
        self._output_specs = self.compute_output_specs(inputs)
    def compute_output_specs(self, input_shape, training=None):
        feature_shape, box_shape = input_shape
        batch_size, num_boxes = box_shape[0:2]
        feature_size = list(feature_shape.values())[0][-1]
        return [batch_size, num_boxes, self.output_size, self.output_size, feature_size]