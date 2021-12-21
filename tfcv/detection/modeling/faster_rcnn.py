import tensorflow as tf

from tfcv.config import config as cfg

from tfcv.classification.modeling.resnet import ResNet

from tfcv.detection.modeling import anchors
from tfcv.detection.modeling.fpn import FPN
from tfcv.detection.modeling.heads import RPNHead, BoxHead, MaskHead
from tfcv.detection.ops import roi_ops, spatial_transform_ops, postprocess_ops, training_ops

class FasterRCNN(tf.keras.Model):
    
    def __init__(self, name='faster_rcnn', *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)

        self.backbone = ResNet(
            cfg.backbone.resnet_id,
            input_shape=[832, 1344, 3],
            freeze_at=0 if cfg.from_scratch else 2,
            freeze_bn=False if cfg.from_scratch else True,
            include_top=False)

        self.fpn = FPN(
            self.backbone.output,
            min_level=cfg.min_level,
            max_level=cfg.max_level)

        self.rpn_head = RPNHead(
            name="rpn_head",
            num_anchors=len(cfg.anchor.aspect_ratios * cfg.anchor.num_scales),
            num_filters=256
        )

        self.box_head = BoxHead(
            num_classes=cfg.num_classes,
            mlp_head_dim=cfg.frcnn.mlp_head_dim,
        )
        if cfg.include_mask:
            self.mask_head = MaskHead(
                num_classes=cfg.num_classes,
                mrcnn_resolution=cfg.mrcnn.resolution,
                name="mask_head"
            )
        else:
            self.mask_head = None

    def call(
        self, 
        images,
        image_info,
        gt_boxes=None,
        gt_classes=None,
        cropped_gt_masks=None,
        training=None):
        _, image_height, image_width, _ = images.get_shape().as_list()

        outputs = dict()

        all_anchors = anchors.Anchors(cfg.min_level, cfg.max_level,
                                    cfg.anchor.num_scales, cfg.anchor.aspect_ratios,
                                    cfg.anchor.scale,
                                    (image_height, image_width))

        backbone_feats = self.backbone(images, training=training)

        fpn_feats = self.fpn(backbone_feats, training=training)

        def rpn_head_fn(features, min_level=2, max_level=6):
            """Region Proposal Network (RPN) for Mask-RCNN."""
            scores_outputs = dict()
            box_outputs = dict()

            for level in range(min_level, max_level + 1):
                scores_outputs[level], box_outputs[level] = self.rpn_head(features[level], training=training)

            return scores_outputs, box_outputs

        rpn_score_outputs, rpn_box_outputs = rpn_head_fn(
            features=fpn_feats,
            min_level=cfg.min_level,
            max_level=cfg.max_level
        )

        if training:
            rpn_pre_nms_topn = cfg.rpn.train.pre_nms_topn
            rpn_post_nms_topn = cfg.rpn.train.post_nms_topn
            rpn_nms_threshold = cfg.rpn.train.nms_threshold

        else:
            rpn_pre_nms_topn = cfg.rpn.test.pre_nms_topn
            rpn_post_nms_topn = cfg.rpn.test.post_nms_topn
            rpn_nms_threshold = cfg.rpn.test.nms_thresh

        rpn_box_scores, rpn_box_rois = roi_ops.multilevel_propose_rois(
            scores_outputs=rpn_score_outputs,
            box_outputs=rpn_box_outputs,
            all_anchors=all_anchors,
            image_info=image_info,
            rpn_pre_nms_topn=rpn_pre_nms_topn,
            rpn_post_nms_topn=rpn_post_nms_topn,
            rpn_nms_threshold=rpn_nms_threshold,
            rpn_min_size=cfg.rpn.min_size,
            bbox_reg_weights=None
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
                batch_size_per_im=cfg.proposal.batch_size_per_im,
                fg_fraction=cfg.proposal.fg_fraction,
                fg_thresh=cfg.proposal.fg_thresh,
                bg_thresh_hi=cfg.proposal.bg_thresh_hi,
                bg_thresh_lo=cfg.proposal.bg_thresh_lo
            )

        # Performs multi-level RoIAlign.
        box_roi_features = spatial_transform_ops.multilevel_crop_and_resize(
            features=fpn_feats,
            boxes=rpn_box_rois,
            output_size=7,
            training=training
        )

        class_outputs, box_outputs, _ = self.box_head(inputs=box_roi_features)

        if not training:
            detections = postprocess_ops.generate_detections_gpu(
                class_outputs=class_outputs,
                box_outputs=box_outputs,
                anchor_boxes=rpn_box_rois,
                image_info=image_info,
                pre_nms_num_detections=cfg.rpn.test.post_nms_topn,
                post_nms_num_detections=cfg.frcnn.test.detections_per_image,
                nms_threshold=cfg.frcnn.test.nms,
                bbox_reg_weights=cfg.frcnn.bbox_reg_weights
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
                bbox_reg_weights=cfg.frcnn.bbox_reg_weights
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
        if not cfg.include_mask:
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
                max_num_fg=int(cfg.proposal.batch_size_per_im * cfg.proposal.fg_fraction)
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
                output_size=cfg.mrcnn.resolution
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
    