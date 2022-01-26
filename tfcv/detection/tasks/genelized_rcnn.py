import tensorflow as tf

from tfcv.ops import preprocess_ops

from tfcv.utils.amp import fp16_to_fp32_nested
from tfcv.ops.losses.mask_rcnn_loss import MaskRCNNLoss
from tfcv.ops.losses.fast_rcnn_loss import FastRCNNLoss
from tfcv.ops.losses.rpn_loss import RPNLoss
from tfcv.detection.tasks.base import DetectionTask

class GenelizedRCNNTask(DetectionTask):

    def train_forward(self, model, inputs):
        model_outputs = model(
            images=inputs['images'],
            image_info=inputs['image_info'],
            gt_boxes=inputs['gt_boxes'],
            gt_classes=inputs['gt_classes'],
            cropped_gt_masks=inputs['cropped_gt_masks'] if self._params.include_mask else None,
            training=True)
        model_outputs = fp16_to_fp32_nested(model_outputs)

        losses = self._build_loss(model_outputs, inputs)
        losses['l2_regularization_loss'] = tf.add_n([
            tf.nn.l2_loss(tf.cast(v, dtype=tf.float32))
            for v in model.trainable_variables
            if not any([pattern in v.name for pattern in ["batch_normalization", "bias", "beta"]])
        ]) * self._params.loss.l2_weight_decay

        raw_loss = tf.math.reduce_sum(list(losses.values()))
        return (raw_loss, losses, None)

    def _build_loss(self, model_outputs, inputs):
        if self._params.include_mask:
            mask_rcnn_loss = MaskRCNNLoss()(model_outputs, inputs)
            mask_rcnn_loss *= self._params.loss.mask_weight

        fast_rcnn_class_loss, fast_rcnn_box_loss = FastRCNNLoss(self._params.num_classes)(model_outputs, inputs)
        fast_rcnn_box_loss *= self._params.loss.fast_rcnn_box_weight

        rpn_score_loss, rpn_box_loss = RPNLoss(
            batch_size=self._params.train_batch_size,
            rpn_batch_size_per_im=self._params.rpn.batch_size_per_im,
            min_level=self._params.min_level,
            max_level=self._params.max_level)(model_outputs, inputs)
        rpn_box_loss *= self._params.loss.rpn_box_weight

        losses = {
            'fast_rcnn_class_loss': fast_rcnn_class_loss,
            'fast_rcnn_box_loss': fast_rcnn_box_loss,
            'rpn_score_loss': rpn_score_loss,
            'rpn_box_loss': rpn_box_loss
        }
        if self._params.include_mask:
            losses['mask_rcnn_loss'] = mask_rcnn_loss
        return losses
        
    def inference_forward(self, model, inputs):
        detections = model(
            images=inputs['images'],
            image_info=inputs['image_info'],
            training=False)
        detections['image_info'] = inputs['image_info']
        if 'source_ids' in inputs:
            detections['source_ids'] = inputs['source_ids']
        return detections
    
    def preprocess(self, image):
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = preprocess_ops.normalize_image(image, self._params.data.pixel_std, self._params.data.pixel_mean)
        
        image, image_info, _, _ = preprocess_ops.resize_and_pad(
            image=image,
            target_size=self._params.data.image_size,
            stride=2 ** self._params.max_level,
            boxes=None,
            masks=None
        )
        return {'images': image, 'image_info': image_info}

        
