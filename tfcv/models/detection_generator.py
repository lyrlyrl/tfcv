import tensorflow as tf

from tfcv.layers.base import Layer
from tfcv.evaluate import box_utils
class DetectionGenerator(Layer):

    default_name = 'detection_generator'

    def __init__(
        self, 
        pre_nms_num_detections=1000,
        post_nms_num_detections=100,
        nms_threshold=0.3,
        nms_score_threshold=0.0,
        bbox_reg_weights=(10., 10., 5., 5.),
        name=None):
        self._init(locals())
        super(DetectionGenerator, self).__init__(trainable=False, name=name)
    
    def call(self, 
            class_outputs,
            box_outputs,
            anchor_boxes,
            image_info,
            training=None):
        with tf.name_scope('generate_detections'):

            batch_size, num_boxes, num_classes = class_outputs.get_shape().as_list()
            softmax_class_outputs = tf.nn.softmax(class_outputs)

            # Remove background
            scores = tf.slice(softmax_class_outputs, [0, 0, 1], [-1, -1, -1])
            boxes = tf.slice(
                tf.reshape(box_outputs, [batch_size, num_boxes, num_classes, 4]),
                [0, 0, 1, 0], [-1, -1, -1, -1]
            )

            anchor_boxes = tf.expand_dims(anchor_boxes, axis=2) * tf.ones([1, 1, num_classes - 1, 1])

            num_detections = num_boxes * (num_classes - 1)

            boxes = tf.reshape(boxes, [batch_size, num_detections, 4])
            scores = tf.reshape(scores, [batch_size, num_detections, 1])
            anchor_boxes = tf.reshape(anchor_boxes, [batch_size, num_detections, 4])

            # Decode
            boxes = box_utils.decode_boxes(boxes, anchor_boxes, self.bbox_reg_weights)

            # Clip boxes
            height = tf.expand_dims(image_info[:, 0:1], axis=-1)
            width = tf.expand_dims(image_info[:, 1:2], axis=-1)
            boxes = box_utils.clip_boxes(boxes, height, width)

            # NMS
            pre_nms_boxes = box_utils.to_normalized_coordinates(boxes, height, width)
            pre_nms_boxes = tf.reshape(pre_nms_boxes, [batch_size, num_boxes, num_classes - 1, 4])
            pre_nms_scores = tf.reshape(scores, [batch_size, num_boxes, num_classes - 1])

            # fixed problems when running with Keras AMP
            pre_nms_boxes = tf.cast(pre_nms_boxes, dtype=tf.float32)
            pre_nms_scores = tf.cast(pre_nms_scores, dtype=tf.float32)

            post_nms_boxes, post_nms_scores, post_nms_classes, \
            post_nms_num_valid_boxes = tf.image.combined_non_max_suppression(
                pre_nms_boxes,
                pre_nms_scores,
                max_output_size_per_class=self.pre_nms_num_detections,
                max_total_size=self.post_nms_num_detections,
                iou_threshold=self.nms_threshold,
                score_threshold=self.nms_score_threshold,
                pad_per_class=False
            )

            post_nms_classes = post_nms_classes + 1

            post_nms_boxes = box_utils.to_absolute_coordinates(post_nms_boxes, height, width)

        return post_nms_num_valid_boxes, post_nms_boxes, tf.cast(post_nms_classes, dtype=tf.float32), post_nms_scores
    
    def _build(self, batch_size, training=None):
        self._output_specs = self.compute_output_specs(batch_size)
    
    def compute_output_specs(self, batch_size):
        return {
            'num_detections': [batch_size],
            'detection_boxes': [batch_size, self.post_nms_num_detections, 4],
            'detection_classes': [batch_size, self.post_nms_num_detections],
            'detection_scores': [batch_size, self.post_nms_num_detections],
        }