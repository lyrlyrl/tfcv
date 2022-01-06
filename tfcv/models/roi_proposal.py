from tfcv.layers.base import Layer

from tfcv.ops import training_ops

class RoiProposal(Layer):
    default_name = 'proposal'
    def __init__(
        self, 
        batch_size_per_im,
        fg_fraction,
        fg_thresh,
        bg_thresh_hi,
        bg_thresh_lo,
        name=None):
        self._init(locals())
        super().__init__(trainable=False, name=name)
    
    def call(self, rpn_box_rois, gt_boxes, gt_classes, training=None):
        box_targets, class_targets, rpn_box_rois, proposal_to_label_map = training_ops.proposal_label_op(
            rpn_box_rois,
            gt_boxes,
            gt_classes,
            batch_size_per_im=self.batch_size_per_im,
            fg_fraction=self.fg_fraction,
            fg_thresh=self.fg_thresh,
            bg_thresh_hi=self.bg_thresh_hi,
            bg_thresh_lo=self.bg_thresh_lo
        )
        return (box_targets, class_targets, rpn_box_rois, proposal_to_label_map)
    
    def compute_output_specs(self, input_shape):
        batch_size = input_shape[0]

        return (
            [batch_size, self.batch_size_per_im, 4],
            [batch_size, self.batch_size_per_im,],
            [batch_size, self.batch_size_per_im, 4],
            [batch_size, self.batch_size_per_im],
        )