import tensorflow as tf

from tfcv.layers.base import Layer
from tfcv.layers.utils import need_build
from tfcv.models.heads import FCBoxHead, MultilevelRPNHead, MaskHead
from tfcv.models.roi_generator import RoiGenerator
from tfcv.models.roi_aligner import RoiAligner
from tfcv.models.roi_proposal import RoiProposal
from tfcv.models.detection_generator import DetectionGenerator
from tfcv.models.fpn import FPN
from tfcv.models.resnet import ResNet
from tfcv.ops import training_ops, anchors

class GenelizedRCNN(Layer):

    default_name = 'genelized_rcnn'

    def __init__(
        self, 
        cfg, 
        trainable=True,
        name=None):
        self._init(locals())
        super(GenelizedRCNN, self).__init__(trainable=trainable, name=name)
        self._init_layers()

    def _init_layers(self):
        self._init_backbone()
        self._init_box_layers()
        if self.cfg.include_mask:
            self._init_mask_layers()

    def _init_backbone(self):
        self._layers['backbone'] = ResNet(
            self.cfg.backbone.resnet_id,
            include_top=False,
            trainable=self.trainable)

    def _init_box_layers(self):
        self._layers['fpn'] = FPN(
            min_level=self.cfg.min_level,
            max_level=self.cfg.max_level,
            num_filters=self.cfg.fpn.num_filters,
            trainable=self.trainable)
        self._layers['rpn_head'] = MultilevelRPNHead(
            min_level=self.cfg.min_level,
            max_level=self.cfg.max_level,
            num_anchors=len(self.cfg.anchor.aspect_ratios * self.cfg.anchor.num_scales),
            num_filters=self.cfg.rpn.num_filters,
            name='rpn_head',
            trainable=self.trainable)
        self._layers['roi_generator'] = RoiGenerator(
            train_pre_nms_topn = self.cfg.rpn.train.pre_nms_topn,
            train_post_nms_topn = self.cfg.rpn.train.post_nms_topn,
            train_nms_threshold = self.cfg.rpn.train.nms_threshold,
            test_pre_nms_topn = self.cfg.rpn.test.pre_nms_topn,
            test_post_nms_topn = self.cfg.rpn.test.post_nms_topn,
            test_nms_threshold = self.cfg.rpn.test.nms_threshold,
            min_size = self.cfg.rpn.min_size
        )
        self._layers['roi_aligner'] = RoiAligner(
            output_size=7
        )
        self._layers['proposal'] = RoiProposal(
            batch_size_per_im=self.cfg.proposal.batch_size_per_im,
            fg_fraction=self.cfg.proposal.fg_fraction,
            fg_thresh=self.cfg.proposal.fg_thresh,
            bg_thresh_hi=self.cfg.proposal.bg_thresh_hi,
            bg_thresh_lo=self.cfg.proposal.bg_thresh_lo,
            name = 'proposal'
        )
        self._layers['box_head'] = FCBoxHead(
            num_classes=self.cfg.num_classes,
            mlp_head_dim=self.cfg.frcnn.mlp_head_dim,
            name='box_head',
            trainable=self.trainable
        )
        self._layers['detection_generator'] = DetectionGenerator(
            pre_nms_num_detections=self.cfg.rpn.test.post_nms_topn,
            post_nms_num_detections=self.cfg.frcnn.test.detections_per_image,
            nms_threshold=self.cfg.frcnn.test.nms,
            nms_score_threshold=self.cfg.frcnn.test.score,
            bbox_reg_weights=self.cfg.frcnn.bbox_reg_weights
        )

    def _init_mask_layers(self):
        self._layers['mask_head'] = MaskHead(
            num_classes=self.cfg.num_classes,
            mrcnn_resolution=self.cfg.mrcnn.resolution,
            trainable=self.trainable
        )
        self._layers['mask_roi_aligner'] = RoiAligner(
            output_size=14
        )

    @need_build
    def call(
        self, 
        images,
        image_info,
        gt_classes=None,
        gt_boxes=None,
        cropped_gt_masks=None, 
        training=None):

        backbone_feats = self._layers['backbone'](images, training=training)

        fpn_feats = self._layers['fpn'](backbone_feats, training=training)

        rpn_score_outputs, rpn_box_outputs = self._layers['rpn_head'](
            fpn_feats, training=training
        )
        
        _, rpn_box_rois = self._layers['roi_generator'](
            scores_outputs=rpn_score_outputs,
            box_outputs=rpn_box_outputs,
            anchor_boxes=self.anchor_boxes,
            image_info=image_info
        )

        rpn_box_rois = tf.cast(rpn_box_rois, dtype=tf.float32)

        if training:
            rpn_box_rois = tf.stop_gradient(rpn_box_rois)

            # Sampling
            box_targets, class_targets, rpn_box_rois, proposal_to_label_map = self._layers['proposal'](
                rpn_box_rois,
                gt_boxes,
                gt_classes
            )

        box_roi_features = self._layers['roi_aligner'](
            features=fpn_feats,
            boxes=rpn_box_rois,
            training=training
        )

        class_outputs, box_outputs = self._layers['box_head'](box_roi_features, training=training)

        if not training:
            detections = self._layers['detection_generator'](
                class_outputs=class_outputs,
                box_outputs=box_outputs,
                anchor_boxes=rpn_box_rois,
                image_info=image_info,
            )

            outputs = {
                'num_detections': detections[0],
                'detection_boxes': detections[1],
                'detection_classes': detections[2],
                'detection_scores': detections[3],
            }
        else:
            encoded_box_targets = training_ops.encode_box_targets(
                boxes=rpn_box_rois,
                gt_boxes=box_targets,
                gt_labels=class_targets,
                bbox_reg_weights=self.cfg.frcnn.bbox_reg_weights
            )

            outputs = {
                'rpn_score_outputs': rpn_score_outputs,
                'rpn_box_outputs': rpn_box_outputs,
                'class_outputs': class_outputs,
                'box_outputs': box_outputs,
                'class_targets': class_targets,
                'box_targets': encoded_box_targets,
                'box_rois': rpn_box_rois,
            }

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

        mask_roi_features = self._layers['mask_roi_aligner'](
            features=fpn_feats,
            boxes=selected_box_rois,
            training=training
        )

        mask_outputs = self._layers['mask_head'](
            mask_roi_features, 
            class_indices,
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

    def _build(self, input_shape):
        batch_size, image_height, image_width, _ = input_shape
        self.anchor_boxes = anchors.Anchors(
            self.cfg.min_level, 
            self.cfg.max_level,
            self.cfg.anchor.num_scales, 
            self.cfg.anchor.aspect_ratios,
            self.cfg.anchor.scale,
            (image_height, image_width)
        ).get_unpacked_boxes()
        with tf.name_scope(self.name):
            self._layers['backbone'].build(input_shape)
            self._layers['fpn'].build(self._layers['backbone'].output_specs)
            self._layers['rpn_head'].build(self._layers['fpn'].output_specs)
            self._output_specs = self._layers['rpn_head'].output_specs
            self._layers['box_head'].build(
                [batch_size, None] +
                [self._layers['roi_aligner'].output_size]*2 +
                [self._layers['fpn'].num_filters])
            if self.cfg.include_mask:
                self._layers['mask_head'].build(
                    [batch_size, None] +
                    [self._layers['mask_roi_aligner'].output_size] * 2 +
                    [self._layers['fpn'].num_filters])
    
    def compute_output_specs(self, input_shape):
        pass
            
