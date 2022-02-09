import tensorflow as tf

from tfcv.ops import preprocess_ops, anchors
from tfcv.losses import focal_loss

from tfcv.datasets.coco.decoder import COCOExampleDecoder
from tfcv.utils.amp import fp16_to_fp32_nested

from tfcv.detection.tasks.base import DetectionTask
from tfcv.detection.tasks.utils import multi_level_flatten

class RetinanetTask(DetectionTask):

    def train_forward(self, model, inputs):
        images, labels = inputs

        model_outputs = model(images, training=True)
        model_outputs = fp16_to_fp32_nested(model_outputs)

        losses = self._build_loss(model_outputs, labels)
        losses['l2_regularization_loss'] = tf.add_n([
            tf.nn.l2_loss(tf.cast(v, dtype=tf.float32))
            for v in model.trainable_variables
            if not any([pattern in v.name for pattern in ["batch_normalization", "bias", "beta"]])
        ]) * self._params.loss.l2_weight_decay

        raw_loss = tf.math.reduce_sum(list(losses.values()))
        return (raw_loss, losses, None)
    
    def _build_loss(self, outputs, labels):
        cls_sample_weight = labels['cls_weights']
        box_sample_weight = labels['box_weights']
        num_positives = tf.reduce_sum(box_sample_weight) + 1.0
        cls_sample_weight = cls_sample_weight / num_positives
        box_sample_weight = box_sample_weight / num_positives
        
        cls_loss = self._build_cls_loss(outputs['cls_outputs'], labels['cls_targets'], cls_sample_weight)
        box_loss = self._build_box_loss(outputs['box_outputs'], labels['box_targets'], box_sample_weight)
        box_loss = self._params.losses.box_loss_weight * box_loss

        return {'cls_loss': cls_loss, 'box_loss': box_loss}

    def _build_cls_loss(self, outputs, targets, weights):
        y_true_cls = multi_level_flatten(
            targets, last_dim=None)
        y_true_cls = tf.one_hot(y_true_cls, self._params.num_classes)
        y_pred_cls = multi_level_flatten(
            outputs, last_dim=self._params.num_classes)
        
        cls_loss = focal_loss(
            y_true_cls,
            y_pred_cls,
            self._params.losses.focal_loss_alpha,
            self._params.losses.focal_loss_gamma,
        )

        return cls_loss
    
    def _build_box_loss(self, outputs, targets, weights):
        y_true_box = multi_level_flatten(
            targets, last_dim=4)
        y_pred_box = multi_level_flatten(
            outputs, last_dim=4)
        huber_loss = tf.keras.losses.Huber(
            self._params.losses.huber_loss_delta)
        box_loss = huber_loss(
            y_true_box,
            y_pred_box,
            weights
        )
        return box_loss
    
    def train_preprocess(self, data, need_decode=True):
        if need_decode:
            decoder = COCOExampleDecoder(
                use_instance_mask=False,
                include_source_id=False)
            with tf.name_scope('decode'):
                data = decoder.decode(data)
        image = tf.image.convert_image_dtype(data['image'], dtype=tf.float32)
        
        image = preprocess_ops.normalize_image(image, self._params.data.pixel_std, self._params.data.pixel_mean)

        boxes = data['groundtruth_boxes']

        classes = data['groundtruth_classes']
        classes = tf.reshape(tf.cast(classes, dtype=tf.float32), [-1, 1])
        if not self._params.data.use_category:
            classes = tf.cast(tf.greater(classes, 0), dtype=tf.float32)
        
        if self._params.data.skip_crowd_during_training:
            with tf.name_scope('remove_crowded'):
                indices = tf.where(tf.logical_not(data['groundtruth_is_crowd']))
                classes = tf.gather_nd(classes, indices)
                boxes = tf.gather_nd(boxes, indices)
        
        if self._params.data.augment_input:
            image, boxes = preprocess_ops.random_horizontal_flip(image, boxes=boxes)
        
        image, image_info, boxes, _ = preprocess_ops.resize_and_pad(
            image=image,
            target_size=self._params.data.image_size,
            stride=2 ** self._params.max_level,
            boxes=boxes
        )

        padded_image_size = image.get_shape().as_list()[:2]
        max_num_instances = self._params.detections_per_image

        input_anchors = anchors.Anchors(
            self._params.min_level,
            self._params.max_level,
            self._params.anchor.num_scales,
            self._params.anchor.aspect_ratios,
            self._params.anchor.scale,
            padded_image_size
        )

        