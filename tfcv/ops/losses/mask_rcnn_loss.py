import tensorflow as tf

from tfcv.ops.losses.common import _sigmoid_cross_entropy

class MaskRCNNLoss:
    """
    Layer that computes the mask loss of Mask-RCNN.

    This layer implements the mask loss of Mask-RCNN. As the `mask_outputs`
    produces `num_classes` masks for each RoI, the reference model expands
    `mask_targets` to match the shape of `mask_outputs` and selects only the
    target that the RoI has a maximum overlap.
    (Reference: https://github.com/facebookresearch/Detectron/blob/master/detectron/roi_data/mask_rcnn.py)
    Instead, this implementation selects the `mask_outputs` by the `class_targets`
    so that it doesn't expand `mask_targets`.
    """

    def __init__(self):
        # super().__init__(reduction=tf.keras.losses.Reduction.NONE, name=None)
        pass

    def __call__(self, model_outputs, inputs, **kwargs):
        """
        Args:
            inputs: dictionary with model outputs, which has to include:
                mask_outputs: a float tensor representing the prediction for each mask,
                    with a shape of [batch_size, num_masks, mask_height, mask_width].
                mask_targets: a float tensor representing the binary mask of ground truth
                    labels for each mask with a shape of [batch_size, num_masks, mask_height, mask_width].
                select_class_targets: a tensor with a shape of [batch_size, num_masks],
                    representing the foreground mask targets.
        Returns:
            mask_loss: a float tensor representing total mask loss.
        """
        mask_outputs = model_outputs['mask_outputs']
        mask_targets = model_outputs['mask_targets']
        select_class_targets = model_outputs['selected_class_targets']

        batch_size, num_masks, mask_height, mask_width = mask_outputs.get_shape().as_list()

        weights = tf.tile(
            tf.reshape(tf.greater(select_class_targets, 0), [batch_size, num_masks, 1, 1]),
            [1, 1, mask_height, mask_width]
        )
        weights = tf.cast(weights, tf.float32)

        return _sigmoid_cross_entropy(
            multi_class_labels=mask_targets,
            logits=mask_outputs,
            weights=weights,
            sum_by_non_zeros_weights=True
        )

