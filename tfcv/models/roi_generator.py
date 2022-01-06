from tfcv.layers.base import Layer

from tfcv.ops import roi_ops

class RoiGenerator(Layer):
    
    default_name = 'roi_generator'

    def __init__(
        self, 
        train_pre_nms_topn,
        train_post_nms_topn,
        train_nms_threshold,
        test_pre_nms_topn,
        test_post_nms_topn,
        test_nms_threshold,
        min_size=0.0,
        name=None):
        self._init(locals())
        super().__init__(trainable=False, name=name)
    def call(self, 
            scores_outputs, 
            box_outputs, 
            anchor_boxes, 
            image_info, training=None):
        if training:
            rpn_pre_nms_topn = self.train_pre_nms_topn
            rpn_post_nms_topn = self.train_post_nms_topn
            rpn_nms_threshold = self.train_nms_threshold
        else:
            rpn_pre_nms_topn = self.test_pre_nms_topn
            rpn_post_nms_topn = self.test_post_nms_topn
            rpn_nms_threshold = self.test_nms_threshold
        rpn_box_scores, rpn_box_rois = roi_ops.multilevel_propose_rois(
            scores_outputs=scores_outputs,
            box_outputs=box_outputs,
            anchor_boxes=anchor_boxes,
            image_info=image_info,
            rpn_pre_nms_topn=rpn_pre_nms_topn,
            rpn_post_nms_topn=rpn_post_nms_topn,
            rpn_nms_threshold=rpn_nms_threshold,
            rpn_min_size=self.min_size,
            bbox_reg_weights=None,
        )
        return (rpn_box_scores, rpn_box_rois)
    def compute_output_specs(self, input_shape, training=True):
        batch_size = input_shape[0].values()[0][0]
        post_nms_topn = self.train_post_nms_topn if training else self.test_post_nms_topn
        return (
            [batch_size, post_nms_topn], [batch_size, post_nms_topn, 4]
        )