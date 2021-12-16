import tensorflow as tf

from tfcv.detection.losses.common import _softmax_cross_entropy, _huber_loss

class FastRCNNLoss:
    """
    Layer that computes the box and class loss (Fast-RCNN branch) of Mask-RCNN.

    This layer implements the classification and box regression loss of the
    Fast-RCNN branch in Mask-RCNN. As the `box_outputs` produces `num_classes`
    boxes for each RoI, the reference model expands `box_targets` to match the
    shape of `box_outputs` and selects only the target that the RoI has a maximum
    overlap.
    (Reference: https://github.com/facebookresearch/Detectron/blob/master/detectron/roi_data/fast_rcnn.py)
    Instead, this function selects the `box_outputs` by the `class_targets` so
    that it doesn't expand `box_targets`.

    The loss computation has two parts: (1) classification loss is softmax on all
    RoIs. (2) box loss is smooth L1-loss on only positive samples of RoIs.
    Reference: https://github.com/facebookresearch/Detectron/blob/master/detectron/modeling/fast_rcnn_heads.py
    """

    def __init__(self, num_classes):
        # super().__init__(reduction=tf.keras.losses.Reduction.NONE, name=None)
        self._num_classes = num_classes

    def __call__(self, model_outputs, inputs, **kwargs):
        """
        Args:
            inputs: dictionary with model outputs, which has to include:
                class_outputs: a float tensor representing the class prediction for each box
                    with a shape of [batch_size, num_boxes, num_classes].
                box_outputs: a float tensor representing the box prediction for each box
                    with a shape of [batch_size, num_boxes, num_classes * 4].
                class_targets: a float tensor representing the class label for each box
                    with a shape of [batch_size, num_boxes].
                box_targets: a float tensor representing the box label for each box
                    with a shape of [batch_size, num_boxes, 4].
        Returns:
            cls_loss: a float tensor representing total class loss.
            box_loss: a float tensor representing total box regression loss.
        """
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
        box_loss = _fast_rcnn_box_loss(
            box_outputs=box_outputs,
            box_targets=box_targets,
            class_targets=class_targets,
            normalizer=1.0
        )

        class_targets = tf.one_hot(class_targets, self._num_classes)
        class_loss = _fast_rcnn_class_loss(
            class_outputs=class_outputs,
            class_targets_one_hot=class_targets,
            normalizer=1.0
        )

        return class_loss, box_loss

def _fast_rcnn_class_loss(class_outputs, class_targets_one_hot, normalizer=1.0):
    """Computes classification loss."""

    with tf.name_scope('fast_rcnn_class_loss'):
        # The loss is normalized by the sum of non-zero weights before additional
        # normalizer provided by the function caller.

        class_loss = _softmax_cross_entropy(onehot_labels=class_targets_one_hot, logits=class_outputs)

        if isinstance(normalizer, tf.Tensor) or normalizer != 1.0:
            class_loss /= normalizer

    return class_loss


def _fast_rcnn_box_loss(box_outputs, box_targets, class_targets, normalizer=1.0, delta=1.):
    """Computes box regression loss."""
    # delta is typically around the mean value of regression target.
    # for instances, the regression targets of 512x512 input with 6 anchors on
    # P2-P6 pyramid is about [0.1, 0.1, 0.2, 0.2].

    with tf.name_scope('fast_rcnn_box_loss'):
        mask = tf.tile(tf.expand_dims(tf.greater(class_targets, 0), axis=2), [1, 1, 4])

        # The loss is normalized by the sum of non-zero weights before additional
        # normalizer provided by the function caller.
        box_loss = _huber_loss(y_true=box_targets, y_pred=box_outputs, weights=mask, delta=delta)

        if isinstance(normalizer, tf.Tensor) or normalizer != 1.0:
            box_loss /= normalizer

    return box_loss
