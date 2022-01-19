import tensorflow as tf

from tfcv.classification.models.resnet import ResNet

from tfcv.ops import anchors
from tfcv.detection.models.fpn import FPN
from tfcv.detection.models.heads import RPNHead, BoxHead, MaskHead
from tfcv.ops import roi_ops, spatial_transform_ops, postprocess_ops, training_ops

class GenelizedRCNN(tf.keras.Model):
    
    def __init__(self, config, name='genelized_rcnn', *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)
        self.cfg = config
        self.backbone = ResNet(
            self.cfg.backbone.resnet_id,
            input_shape=[832, 1344, 3],
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
        batch_size, image_height, image_width, _ = images.get_shape().as_list()
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
    