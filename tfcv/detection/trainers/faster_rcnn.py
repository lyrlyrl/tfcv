import tensorflow as tf

from tfcv.detection.losses.mask_rcnn_loss import MaskRCNNLoss
from tfcv.detection.losses.fast_rcnn_loss import FastRCNNLoss
from tfcv.detection.losses.rpn_loss import RPNLoss

from tfcv.detection.trainers.base import DetectionTrainer

class FasterRCNNTrainer(DetectionTrainer):
    def train_step(self, inputs):
        with tf.GradientTape() as tape:
            model_outputs = self._model(
                images=inputs['images'],
                image_info=inputs['image_info'],
                gt_boxes=inputs['gt_boxes'],
                gt_classes=inputs['gt_classes'],
                cropped_gt_masks=inputs['cropped_gt_masks'] if self._params.include_mask else None,
                training=True)
            # model_outputs = self._model(1, training=True)
            model_outputs = tf.nest.map_structure(
                lambda x: tf.cast(x, tf.float32), model_outputs)
            losses = self._build_loss(model_outputs, inputs)
            losses['l2_regularization_loss'] = tf.add_n([
                tf.nn.l2_loss(tf.cast(v, dtype=tf.float32))
                for v in self._model.trainable_variables
                if not any([pattern in v.name for pattern in ["batch_normalization", "bias", "beta"]])
            ]) * self._params.loss.l2_weight_decay
            raw_loss = tf.math.reduce_sum(list(losses.values()))
            if isinstance(self._optimizer, tf.keras.mixed_precision.LossScaleOptimizer):
                loss = self._optimizer.get_scaled_loss(raw_loss)
            else:
                loss = raw_loss
        trainable_weights = self._model.trainable_weights
        grads = tape.gradient(loss, trainable_weights)
        if isinstance(self._optimizer, tf.keras.mixed_precision.LossScaleOptimizer):
            grads = self._optimizer.get_unscaled_gradients(grads)
        self._optimizer.apply_gradients(list(zip(grads, trainable_weights)))
        self._train_loss.update_state(raw_loss)
        for metric in self._metrics:
            metric.update_state(losses[metric.name])    
    def inference_step(self, inputs):
        detections = self._model(
            images=inputs['images'],
            image_info=inputs['image_info'],
            training=False)
        detections['source_ids'] = inputs['source_ids']
        detections['image_info'] = inputs['image_info']
        return detections
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
