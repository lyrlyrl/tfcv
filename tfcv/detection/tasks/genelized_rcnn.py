import tensorflow as tf

from tfcv.ops import preprocess_ops, anchors
from tfcv.datasets.coco.decoder import COCOExampleDecoder

from tfcv.utils.amp import fp16_to_fp32_nested

from tfcv.losses import huber_loss, softmax_crossentropy, sigmoid_crossentropy
from tfcv.detection.dataset import TFDataset
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
            mask_rcnn_loss = self._build_mrcnn_loss(model_outputs)
            mask_rcnn_loss *= self._params.loss.mask_weight

        fast_rcnn_class_loss, fast_rcnn_box_loss = self._build_frcnn_loss(model_outputs, inputs)
        fast_rcnn_box_loss *= self._params.loss.fast_rcnn_box_weight

        rpn_score_loss, rpn_box_loss = self._build_rpn_loss(model_outputs, inputs)
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
    
    def _build_rpn_loss(self, model_outputs, inputs):
        score_outputs = model_outputs['rpn_score_outputs']
        box_outputs = model_outputs['rpn_box_outputs']

        score_losses = []
        box_losses = []

        score_normalizer = tf.cast(self._params.train_batch_size * self._params.rpn.batch_size_per_im, dtype=tf.float32)

        for level in range(int(self._params.min_level), int(self._params.max_level + 1)):

            score_targets_at_level = inputs['score_targets_%d' % level]
            score_outputs_at_level = score_outputs[str(level)]
            box_targets_at_level = inputs['box_targets_%d' % level]
            box_outputs_at_level = box_outputs[str(level)]

            with tf.name_scope('rpn_score_loss_level%d'.format(level)):
                mask = tf.math.greater_equal(score_targets_at_level, 0)
                mask = tf.cast(mask, dtype=tf.float32)

                score_targets_at_level = tf.maximum(score_targets_at_level, tf.zeros_like(score_targets_at_level))
                score_targets_at_level = tf.cast(score_targets_at_level, dtype=tf.float32)

                assert score_outputs_at_level.dtype == tf.float32
                assert score_targets_at_level.dtype == tf.float32

                score_loss = sigmoid_crossentropy(
                    multi_class_labels=score_targets_at_level,
                    logits=score_outputs_at_level,
                    weights=mask,
                    sum_by_non_zeros_weights=False
                )

                assert score_loss.dtype == tf.float32

                score_loss /= score_normalizer

                assert score_loss.dtype == tf.float32
            score_losses.append(score_loss)

            with tf.name_scope('rpn_box_loss_level%d'.format(level)):
                mask = tf.not_equal(box_targets_at_level, 0.0)
                mask = tf.cast(mask, tf.float32)

                assert mask.dtype == tf.float32

                # The loss is normalized by the sum of non-zero weights before additional
                # normalizer provided by the function caller.
                box_loss = huber_loss(y_true=box_targets_at_level, y_pred=box_outputs_at_level, weights=mask, delta=1.0)

                assert box_loss.dtype == tf.float32

            box_losses.append(box_loss)

        # Sum per level losses to total loss.
        rpn_score_loss = tf.add_n(score_losses)
        rpn_box_loss = tf.add_n(box_losses)

        return rpn_score_loss, rpn_box_loss
    
    def _build_frcnn_loss(self, model_outputs, inputs):
        class_outputs = model_outputs['class_outputs']
        box_outputs = model_outputs['box_outputs']
        class_targets = model_outputs['class_targets']
        box_targets = model_outputs['box_targets']

        class_targets = tf.cast(class_targets, dtype=tf.int32)

        # Selects the box from `box_outputs` based on `class_targets`, with which
        # the box has the maximum overlap.
        batch_size, num_rois, _ = box_outputs.get_shape().as_list()
        box_outputs = tf.reshape(box_outputs, [batch_size, num_rois, self._num_classes, 4])

        box_indices = tf.reshape(
            class_targets +
            tf.tile(tf.expand_dims(tf.range(batch_size) * num_rois * self._num_classes, 1), [1, num_rois]) +
            tf.tile(tf.expand_dims(tf.range(num_rois) * self._num_classes, 0), [batch_size, 1]),
            [-1]
        )

        box_outputs = tf.matmul(
            tf.one_hot(
                box_indices,
                batch_size * num_rois * self._num_classes,
                dtype=box_outputs.dtype
            ),
            tf.reshape(box_outputs, [-1, 4])
        )

        box_outputs = tf.reshape(box_outputs, [batch_size, -1, 4])
        with tf.name_scope('fast_rcnn_box_loss'):
            mask = tf.tile(tf.expand_dims(tf.greater(class_targets, 0), axis=2), [1, 1, 4])
            # The loss is normalized by the sum of non-zero weights before additional
            # normalizer provided by the function caller.
            box_loss = huber_loss(y_true=box_targets, y_pred=box_outputs, weights=mask, delta=1.0)
        
        with tf.name_scope('fast_rcnn_class_loss'):
            class_targets_one_hot = tf.one_hot(class_targets, self._params.num_classes)
            class_loss = softmax_crossentropy(onehot_labels=class_targets_one_hot, logits=class_outputs)

        return class_loss, box_loss
    
    def _build_mrcnn_loss(self, model_outputs):
        mask_outputs = model_outputs['mask_outputs']
        mask_targets = model_outputs['mask_targets']
        select_class_targets = model_outputs['selected_class_targets']

        batch_size, num_masks, mask_height, mask_width = mask_outputs.get_shape().as_list()

        weights = tf.tile(
            tf.reshape(tf.greater(select_class_targets, 0), [batch_size, num_masks, 1, 1]),
            [1, 1, mask_height, mask_width]
        )
        weights = tf.cast(weights, tf.float32)

        return sigmoid_crossentropy(
            multi_class_labels=mask_targets,
            logits=mask_outputs,
            weights=weights,
            sum_by_non_zeros_weights=True
        )
    
    def inference_forward(self, model, inputs):
        detections = model.call(
            images=inputs['images'],
            image_info=inputs['image_info'],
            training=False)
        detections['image_info'] = inputs['image_info']
        if 'source_ids' in inputs:
            detections['source_ids'] = inputs['source_ids']
        return detections
    
    def inference_preprocess(self, image):
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
    
    def train_preprocess(self, data, need_decode=True):
        if need_decode:
            decoder = COCOExampleDecoder(
                use_instance_mask=self._params.include_mask,
                include_source_id=False)
            with tf.name_scope('decode'):
                data = decoder.decode(data)
        
        image = tf.image.convert_image_dtype(data['image'], dtype=tf.float32)
        
        image = preprocess_ops.normalize_image(image, self._params.data.pixel_std, self._params.data.pixel_mean)

        boxes = data['groundtruth_boxes']
        if self._params.include_mask:
            instance_masks = data['groundtruth_instance_masks']
        else:
            instance_masks = None
        classes = data['groundtruth_classes']
        classes = tf.reshape(tf.cast(classes, dtype=tf.float32), [-1, 1])

        if not self._params.data.use_category:
            classes = tf.cast(tf.greater(classes, 0), dtype=tf.float32)
        
        if self._params.data.skip_crowd_during_training:
            with tf.name_scope('remove_crowded'):
                indices = tf.where(tf.logical_not(data['groundtruth_is_crowd']))
                classes = tf.gather_nd(classes, indices)
                boxes = tf.gather_nd(boxes, indices)

                if self._params.include_mask:
                    instance_masks = tf.gather_nd(instance_masks, indices)
        
        if self._params.data.augment_input:
            if self._params.include_mask:
                image, boxes, instance_masks = preprocess_ops.random_horizontal_flip(image, boxes=boxes, masks=instance_masks)
            else:
                image, boxes = preprocess_ops.random_horizontal_flip(image, boxes=boxes)
        
        # Scaling and padding.
        image, image_info, boxes, instance_masks = preprocess_ops.resize_and_pad(
            image=image,
            target_size=self._params.data.image_size,
            stride=2 ** self._params.max_level,
            boxes=boxes,
            masks=instance_masks
        )
        padded_image_size = image.get_shape().as_list()[:2]
        max_num_instances = self._params.frcnn.test.detections_per_image

        if self._params.include_mask:
            gt_mask_size = self._params.mrcnn.gt_mask_size
            
            cropped_gt_masks = preprocess_ops.crop_gt_masks(
                instance_masks=instance_masks,
                boxes=boxes,
                gt_mask_size=gt_mask_size,
                image_size=padded_image_size
            )
            cropped_gt_masks = preprocess_ops.pad_to_fixed_size(
                data=cropped_gt_masks,
                pad_value=-1,
                output_shape=[max_num_instances, (gt_mask_size + 4) ** 2]
            )
            cropped_gt_masks = tf.reshape(
                cropped_gt_masks, 
                [max_num_instances, gt_mask_size + 4, gt_mask_size + 4])
        
        input_anchors = anchors.Anchors(
            self._params.min_level,
            self._params.max_level,
            self._params.anchor.num_scales,
            self._params.anchor.aspect_ratios,
            self._params.anchor.scale,
            padded_image_size
        )

        anchor_labeler = anchors.RpnAnchorLabeler(
            input_anchors,
            self._params.num_classes,
            self._params.rpn.positive_overlap,
            self._params.rpn.negative_overlap,
            self._params.rpn.batch_size_per_im,
            self._params.rpn.fg_fraction
        )

        score_targets, box_targets = anchor_labeler.label_anchors(boxes, classes)

        boxes = preprocess_ops.pad_to_fixed_size(boxes, -1, [max_num_instances, 4])

        classes = preprocess_ops.pad_to_fixed_size(classes, -1, [max_num_instances, 1])
        
        features = {
            'gt_boxes': boxes,
            'gt_classes': classes,
            'images': image,
            'image_info': image_info
        }

        if self._params.include_mask:
            features['cropped_gt_masks'] = cropped_gt_masks

        for level in range(self._params.min_level, self._params.max_level + 1):
            features['score_targets_%d' % level] = score_targets[str(level)]
            features['box_targets_%d' % level] = box_targets[str(level)]
        
        return features
        
    def evaluate_preprocess(self, data, need_decode=True):
        if need_decode:
            decoder = COCOExampleDecoder(
                use_instance_mask=self._params.include_mask,
                include_source_id=True)
            with tf.name_scope('decode'):
                data = decoder.decode(data)

        image = tf.image.convert_image_dtype(data['image'], dtype=tf.float32)
        
        image = preprocess_ops.normalize_image(image, self._params.data.pixel_std, self._params.data.pixel_mean)

        source_id = data['source_id']

        if source_id.dtype == tf.string:
            source_id = tf.cast(tf.strings.to_number(source_id), tf.int64)

        with tf.control_dependencies([source_id]):
            source_id = tf.cond(
                pred=tf.equal(tf.size(input=source_id), 0),
                true_fn=lambda: tf.cast(tf.constant(-1), tf.int64),
                false_fn=lambda: tf.identity(source_id)
            )
        
        image, image_info, _, _ = preprocess_ops.resize_and_pad(
            image=image,
            target_size=self._params.data.image_size,
            stride=2 ** self._params.max_level
        )

        features = {
            'source_id': source_id,
            'images': image,
            'image_info': image_info
        }

        return features
    