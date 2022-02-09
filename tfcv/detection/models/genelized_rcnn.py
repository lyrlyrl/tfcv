import tensorflow as tf

from tfcv.classification.models.resnet import ResNet

from tfcv.ops import anchors
from tfcv.common import expand_image_shape
from tfcv.detection.models.fpn import FPN
from tfcv.ops import roi_ops, spatial_transform_ops, postprocess_ops, training_ops

class RPNHead(tf.keras.layers.Layer):

    def __init__(self, 
                name,
                num_anchors,
                num_filters,
                *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)
        """Shared RPN heads."""

        # TODO(chiachenc): check the channel depth of the first convolution.
        self._conv = tf.keras.layers.Conv2D(
            num_filters,
            kernel_size=3,
            strides=1,
            activation=tf.nn.relu,
            bias_initializer=tf.keras.initializers.Zeros(),
            kernel_initializer=tf.random_normal_initializer(stddev=0.01),
            padding='same',
            name='rpn-conv'
        )

        # Proposal classification scores
        # scores = tf.keras.layers.Conv2D(
        self._classifier = tf.keras.layers.Conv2D(
            num_anchors,
            kernel_size=1,
            strides=1,
            bias_initializer=tf.keras.initializers.Zeros(),
            kernel_initializer=tf.random_normal_initializer(stddev=0.01),
            padding='valid',
            name='rpn-score'
        )

        # Proposal bbox regression deltas
        # bboxes = tf.keras.layers.Conv2D(
        self._box_regressor = tf.keras.layers.Conv2D(
            4 * num_anchors,
            kernel_size=1,
            strides=1,
            bias_initializer=tf.keras.initializers.Zeros(),
            kernel_initializer=tf.random_normal_initializer(stddev=0.01),
            padding='valid',
            name='rpn-box'
        )

    def call(self, inputs, *args, **kwargs):
        net = self._conv(inputs)
        scores = self._classifier(net)
        bboxes = self._box_regressor(net)

        return scores, bboxes


class BoxHead(tf.keras.layers.Layer):

    def __init__(self, num_classes=91, mlp_head_dim=1024, *args, **kwargs):
        """Box and class branches for the Mask-RCNN model.

        Args:
        roi_features: A ROI feature tensor of shape
          [batch_size, num_rois, height_l, width_l, num_filters].
        num_classes: a integer for the number of classes.
        mlp_head_dim: a integer that is the hidden dimension in the fully-connected
          layers.
        """
        super().__init__(*args, **kwargs)

        self._num_classes = num_classes
        self._mlp_head_dim = mlp_head_dim

        self._dense_fc6 = tf.keras.layers.Dense(
            units=mlp_head_dim,
            activation=tf.nn.relu,
            name='fc6'
        )

        self._dense_fc7 = tf.keras.layers.Dense(
            units=mlp_head_dim,
            activation=tf.nn.relu,
            name='fc7'
        )

        self._dense_class = tf.keras.layers.Dense(
            num_classes,
            kernel_initializer=tf.random_normal_initializer(stddev=0.01),
            bias_initializer=tf.keras.initializers.Zeros(),
            name='class-predict'
        )

        self._dense_box = tf.keras.layers.Dense(
            num_classes * 4,
            kernel_initializer=tf.random_normal_initializer(stddev=0.001),
            bias_initializer=tf.keras.initializers.Zeros(),
            name='box-predict'
        )

    def call(self, inputs, **kwargs):
        """
        Returns:
        class_outputs: a tensor with a shape of
          [batch_size, num_rois, num_classes], representing the class predictions.
        box_outputs: a tensor with a shape of
          [batch_size, num_rois, num_classes * 4], representing the box predictions.
        box_features: a tensor with a shape of
          [batch_size, num_rois, mlp_head_dim], representing the box features.
        """

        # reshape inputs before FC.
        batch_size, num_rois, height, width, filters = inputs.get_shape().as_list()
        
        net = tf.reshape(inputs, [batch_size, num_rois, height * width * filters])

        net = self._dense_fc6(net)

        box_features = self._dense_fc7(net)

        class_outputs = self._dense_class(box_features)

        box_outputs = self._dense_box(box_features)

        return class_outputs, box_outputs, box_features


class MaskHead(tf.keras.layers.Layer):

    @staticmethod
    def _get_stddev_equivalent_to_msra_fill(kernel_size, fan_out):
        """Returns the stddev of random normal initialization as MSRAFill."""
        # Reference: https://github.com/pytorch/pytorch/blob/master/caffe2/operators/filler_op.h#L445-L463
        # For example, kernel size is (3, 3) and fan out is 256, stddev is 0.029.
        # stddev = (2/(3*3*256))^0.5 = 0.029
        return (2 / (kernel_size[0] * kernel_size[1] * fan_out)) ** 0.5

    def __init__(
            self,
            num_classes=91,
            mrcnn_resolution=28,
            *args,
            **kwargs
    ):
        """Mask branch for the Mask-RCNN model.

        Args:
        roi_features: A ROI feature tensor of shape
          [batch_size, num_rois, height_l, width_l, num_filters].
        num_classes: an integer for the number of classes.
        mrcnn_resolution: an integer that is the resolution of masks.
        """
        super().__init__(*args, **kwargs)

        self._num_classes = num_classes
        self._mrcnn_resolution = mrcnn_resolution

        self._conv_stage1 = list()
        kernel_size = (3, 3)
        fan_out = 256

        init_stddev = MaskHead._get_stddev_equivalent_to_msra_fill(kernel_size, fan_out)

        for conv_id in range(4):
            self._conv_stage1.append(tf.keras.layers.Conv2D(
                fan_out,
                kernel_size=kernel_size,
                strides=(1, 1),
                padding='same',
                dilation_rate=(1, 1),
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(stddev=init_stddev),
                bias_initializer=tf.keras.initializers.Zeros(),
                name='mask-conv-l%d' % conv_id
            ))

        kernel_size = (2, 2)
        fan_out = 256

        init_stddev = MaskHead._get_stddev_equivalent_to_msra_fill(kernel_size, fan_out)

        self._conv_stage2 = tf.keras.layers.Conv2DTranspose(
            fan_out,
            kernel_size=kernel_size,
            strides=(2, 2),
            padding='valid',
            activation=tf.nn.relu,
            kernel_initializer=tf.random_normal_initializer(stddev=init_stddev),
            bias_initializer=tf.keras.initializers.Zeros(),
            name='conv5-mask'
        )

        kernel_size = (1, 1)
        fan_out = self._num_classes

        init_stddev = MaskHead._get_stddev_equivalent_to_msra_fill(kernel_size, fan_out)

        self._conv_stage3 = tf.keras.layers.Conv2D(
            fan_out,
            kernel_size=kernel_size,
            strides=(1, 1),
            padding='valid',
            kernel_initializer=tf.random_normal_initializer(stddev=init_stddev),
            bias_initializer=tf.keras.initializers.Zeros(),
            name='mask_fcn_logits'
        )

    def call(self, inputs, **kwargs):
        """
        Args:
            inputs: tuple of two tensors:
                mask_roi_features: a Tensor of shape:
                  [batch_size, num_boxes, output_size, output_size, num_filters].
                class_indices: a Tensor of shape [batch_size, num_rois], indicating
                  which class the ROI is.
            training: whether to build the model for training (or inference).
        Returns:
            mask_outputs: a tensor with a shape of
              [batch_size, num_masks, mask_height, mask_width],
              representing the mask predictions.
            fg_gather_indices: a tensor with a shape of [batch_size, num_masks, 2],
              representing the fg mask targets.
        Raises:
            ValueError: If boxes is not a rank-3 tensor or the last dimension of
              boxes is not 4.
        """
        mask_roi_features, class_indices = inputs
        indices_dtype = tf.int32
        # fixed problems when running with Keras AMP
        class_indices = tf.cast(class_indices, dtype=indices_dtype)

        batch_size, num_rois, height, width, filters = mask_roi_features.get_shape().as_list()

        net = tf.reshape(mask_roi_features, [-1, height, width, filters])

        for conv_id in range(4):
            net = self._conv_stage1[conv_id](net)

        net = self._conv_stage2(net)

        mask_outputs = self._conv_stage3(net)

        mask_outputs = tf.reshape(
            mask_outputs,
            [-1, num_rois, self._mrcnn_resolution, self._mrcnn_resolution, self._num_classes]
        )

        with tf.name_scope('masks_post_processing'):

            mask_outputs = tf.transpose(a=mask_outputs, perm=[0, 1, 4, 2, 3])

            if batch_size == 1:
                indices = tf.reshape(
                    tf.reshape(
                        tf.range(num_rois, dtype=indices_dtype),
                        [batch_size, num_rois, 1]
                    ) * self._num_classes + tf.expand_dims(class_indices, axis=-1),
                    [batch_size, -1]
                )

                mask_outputs = tf.gather(
                    tf.reshape(mask_outputs,
                               [batch_size, -1, self._mrcnn_resolution, self._mrcnn_resolution]),
                    indices,
                    axis=1
                )

                mask_outputs = tf.squeeze(mask_outputs, axis=1)
                mask_outputs = tf.reshape(
                    mask_outputs,
                    [batch_size, num_rois, self._mrcnn_resolution, self._mrcnn_resolution])

            else:
                batch_indices = (
                        tf.expand_dims(tf.range(batch_size, dtype=indices_dtype), axis=1) *
                        tf.ones([1, num_rois], dtype=indices_dtype)
                )

                mask_indices = (
                        tf.expand_dims(tf.range(num_rois, dtype=indices_dtype), axis=0) *
                        tf.ones([batch_size, 1], dtype=indices_dtype)
                )

                gather_indices = tf.stack([batch_indices, mask_indices, class_indices], axis=2)

                mask_outputs = tf.gather_nd(mask_outputs, gather_indices)

        return mask_outputs


class GenelizedRCNN(tf.keras.Model):
    
    def __init__(self, config, name='genelized_rcnn', *args, **kwargs):
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
            max_level=self.cfg.max_level)

        self.rpn_head = RPNHead(
            name="rpn_head",
            num_anchors=len(self.cfg.anchor.aspect_ratios * self.cfg.anchor.num_scales),
            num_filters=256
        )

        self.box_head = BoxHead(
            num_classes=self.cfg.num_classes,
            mlp_head_dim=self.cfg.frcnn.mlp_head_dim,
        )
        if self.cfg.include_mask:
            self.mask_head = MaskHead(
                num_classes=self.cfg.num_classes,
                mrcnn_resolution=self.cfg.mrcnn.resolution,
                name="mask_head"
            )
        else:
            self.mask_head = None

    def call(
        self, 
        images,
        image_info,
        anchor_boxes=None,
        gt_boxes=None,
        gt_classes=None,
        cropped_gt_masks=None,
        training=None):
        _, image_height, image_width, _ = images.get_shape().as_list()
        outputs = dict()
        if not anchor_boxes:
            all_anchors = anchors.Anchors(self.cfg.min_level, self.cfg.max_level,
                                    self.cfg.anchor.num_scales, self.cfg.anchor.aspect_ratios,
                                    self.cfg.anchor.scale,
                                    (image_height, image_width))
            anchor_boxes = all_anchors.get_unpacked_boxes()

        backbone_feats = self.backbone(images, training=training)

        fpn_feats = self.fpn(backbone_feats, training=training)

        def rpn_head_fn(features, min_level=2, max_level=6):
            """Region Proposal Network (RPN) for Mask-RCNN."""
            scores_outputs = dict()
            box_outputs = dict()

            for level in range(min_level, max_level + 1):
                scores_outputs[str(level)], box_outputs[str(level)] = self.rpn_head(features[str(level)], training=training)

            return scores_outputs, box_outputs

        rpn_score_outputs, rpn_box_outputs = rpn_head_fn(
            features=fpn_feats,
            min_level=self.cfg.min_level,
            max_level=self.cfg.max_level
        )

        if training:
            rpn_pre_nms_topn = self.cfg.rpn.train.pre_nms_topn
            rpn_post_nms_topn = self.cfg.rpn.train.post_nms_topn
            rpn_nms_threshold = self.cfg.rpn.train.nms_threshold

        else:
            rpn_pre_nms_topn = self.cfg.rpn.test.pre_nms_topn
            rpn_post_nms_topn = self.cfg.rpn.test.post_nms_topn
            rpn_nms_threshold = self.cfg.rpn.test.nms_thresh
        rpn_box_scores, rpn_box_rois = roi_ops.multilevel_propose_rois(
            scores_outputs=rpn_score_outputs,
            box_outputs=rpn_box_outputs,
            anchor_boxes=anchor_boxes,
            image_info=image_info,
            rpn_pre_nms_topn=rpn_pre_nms_topn,
            rpn_post_nms_topn=rpn_post_nms_topn,
            rpn_nms_threshold=rpn_nms_threshold,
            rpn_min_size=self.cfg.rpn.min_size,
            bbox_reg_weights=None,
        )

        rpn_box_rois = tf.cast(rpn_box_rois, dtype=tf.float32)

        if training:
            rpn_box_rois = tf.stop_gradient(rpn_box_rois)
            rpn_box_scores = tf.stop_gradient(rpn_box_scores)  # TODO Jonathan: Unused => Shall keep ?

            # Sampling
            box_targets, class_targets, rpn_box_rois, proposal_to_label_map = training_ops.proposal_label_op(
                rpn_box_rois,
                gt_boxes,
                gt_classes,
                batch_size_per_im=self.cfg.proposal.batch_size_per_im,
                fg_fraction=self.cfg.proposal.fg_fraction,
                fg_thresh=self.cfg.proposal.fg_thresh,
                bg_thresh_hi=self.cfg.proposal.bg_thresh_hi,
                bg_thresh_lo=self.cfg.proposal.bg_thresh_lo
            )

        # Performs multi-level RoIAlign.
        box_roi_features = spatial_transform_ops.multilevel_crop_and_resize(
            features=fpn_feats,
            boxes=rpn_box_rois,
            output_size=7,
            training=training
        )
        class_outputs, box_outputs, _ = self.box_head(box_roi_features, training=training)

        if not training:
            detections = postprocess_ops.generate_detections_gpu(
                class_outputs=class_outputs,
                box_outputs=box_outputs,
                anchor_boxes=rpn_box_rois,
                image_info=image_info,
                pre_nms_num_detections=self.cfg.rpn.test.post_nms_topn,
                post_nms_num_detections=self.cfg.frcnn.test.detections_per_image,
                nms_threshold=self.cfg.frcnn.test.nms,
                nms_score_threshold=self.cfg.frcnn.test.score,
                bbox_reg_weights=self.cfg.frcnn.bbox_reg_weights
            )

            outputs.update({
                'num_detections': detections[0],
                'detection_boxes': detections[1],
                'detection_classes': detections[2],
                'detection_scores': detections[3],
            })

        else:  # is training
            encoded_box_targets = training_ops.encode_box_targets(
                boxes=rpn_box_rois,
                gt_boxes=box_targets,
                gt_labels=class_targets,
                bbox_reg_weights=self.cfg.frcnn.bbox_reg_weights
            )

            outputs.update({
                'rpn_score_outputs': rpn_score_outputs,
                'rpn_box_outputs': rpn_box_outputs,
                'class_outputs': class_outputs,
                'box_outputs': box_outputs,
                'class_targets': class_targets,
                'box_targets': encoded_box_targets,
                'box_rois': rpn_box_rois,
            })

        # Faster-RCNN mode.
        if not self.cfg.include_mask:
            return outputs

        # Mask sampling
        if not training:
            selected_box_rois = outputs['detection_boxes']
            class_indices = outputs['detection_classes']

        else:
            selected_class_targets, selected_box_targets, \
            selected_box_rois, proposal_to_label_map = training_ops.select_fg_for_masks(
                class_targets=class_targets,
                box_targets=box_targets,
                boxes=rpn_box_rois,
                proposal_to_label_map=proposal_to_label_map,
                max_num_fg=int(self.cfg.proposal.batch_size_per_im * self.cfg.proposal.fg_fraction)
            )

            class_indices = tf.cast(selected_class_targets, dtype=tf.int32)

        mask_roi_features = spatial_transform_ops.multilevel_crop_and_resize(
            features=fpn_feats,
            boxes=selected_box_rois,
            output_size=14,
            training=training
        )

        mask_outputs = self.mask_head(
            inputs=(mask_roi_features, class_indices),
            training=training
        )

        if training:
            mask_targets = training_ops.get_mask_targets(
                fg_boxes=selected_box_rois,
                fg_proposal_to_label_map=proposal_to_label_map,
                fg_box_targets=selected_box_targets,
                mask_gt_labels=cropped_gt_masks,
                output_size=self.cfg.mrcnn.resolution
            )

            outputs.update({
                'mask_outputs': mask_outputs,
                'mask_targets': mask_targets,
                'selected_class_targets': selected_class_targets,
            })

        else:
            outputs.update({
                'detection_masks': tf.nn.sigmoid(mask_outputs),
            })

        return outputs
    