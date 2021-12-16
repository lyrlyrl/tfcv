import tensorflow as tf

from tfcv.detection.losses.common import _sigmoid_cross_entropy, _huber_loss

def _rpn_score_loss(score_outputs, score_targets, normalizer=1.0):
    """Computes score loss."""

    with tf.name_scope('rpn_score_loss'):

        # score_targets has three values:
        # * (1) score_targets[i]=1, the anchor is a positive sample.
        # * (2) score_targets[i]=0, negative.
        # * (3) score_targets[i]=-1, the anchor is don't care (ignore).

        mask = tf.math.greater_equal(score_targets, 0)
        mask = tf.cast(mask, dtype=tf.float32)

        score_targets = tf.maximum(score_targets, tf.zeros_like(score_targets))
        score_targets = tf.cast(score_targets, dtype=tf.float32)

        assert score_outputs.dtype == tf.float32
        assert score_targets.dtype == tf.float32

        score_loss = _sigmoid_cross_entropy(
            multi_class_labels=score_targets,
            logits=score_outputs,
            weights=mask,
            sum_by_non_zeros_weights=False
        )

        assert score_loss.dtype == tf.float32

        if isinstance(normalizer, tf.Tensor) or normalizer != 1.0:
            score_loss /= normalizer

        assert score_loss.dtype == tf.float32

    return score_loss


def _rpn_box_loss(box_outputs, box_targets, normalizer=1.0, delta=1. / 9):
    """Computes box regression loss."""
    # delta is typically around the mean value of regression target.
    # for instances, the regression targets of 512x512 input with 6 anchors on
    # P2-P6 pyramid is about [0.1, 0.1, 0.2, 0.2].

    with tf.name_scope('rpn_box_loss'):
        mask = tf.not_equal(box_targets, 0.0)
        mask = tf.cast(mask, tf.float32)

        assert mask.dtype == tf.float32

        # The loss is normalized by the sum of non-zero weights before additional
        # normalizer provided by the function caller.
        box_loss = _huber_loss(y_true=box_targets, y_pred=box_outputs, weights=mask, delta=delta)

        assert box_loss.dtype == tf.float32

        if isinstance(normalizer, tf.Tensor) or normalizer != 1.0:
            box_loss /= normalizer

        assert box_loss.dtype == tf.float32

    return box_loss


class RPNLoss:
    """
    Layer that computes total RPN detection loss.

    Computes total RPN detection loss including box and score from all levels.
    """

    def __init__(self, batch_size, rpn_batch_size_per_im, min_level, max_level):
        # super().__init__(reduction=tf.keras.losses.Reduction.NONE, name=None)
        self._batch_size = batch_size
        self._rpn_batch_size_per_im = rpn_batch_size_per_im
        self._min_level = min_level
        self._max_level = max_level

    def __call__(self, model_outputs, inputs, **kwargs):
        """
        Args:
            inputs: dictionary with model outputs, which has to include:
                score_outputs: an OrderDict with keys representing levels and values
                    representing scores in [batch_size, height, width, num_anchors].
                box_outputs: an OrderDict with keys representing levels and values
                    representing box regression targets in [batch_size, height, width, num_anchors * 4].
                score_targets_*: ground truth score targets
                box_targets_*: ground truth box targets
        Returns:
            rpn_score_loss: a float tensor representing total score loss.
            rpn_box_loss: a float tensor representing total box regression loss.
        """
        score_outputs = model_outputs['rpn_score_outputs']
        box_outputs = model_outputs['rpn_box_outputs']

        score_losses = []
        box_losses = []

        for level in range(int(self._min_level), int(self._max_level + 1)):

            score_targets_at_level = inputs['score_targets_%d' % level]
            box_targets_at_level = inputs['box_targets_%d' % level]

            score_losses.append(
                _rpn_score_loss(
                    score_outputs=score_outputs[level],
                    score_targets=score_targets_at_level,
                    normalizer=tf.cast(self._batch_size * self._rpn_batch_size_per_im, dtype=tf.float32)
                )
            )

            box_losses.append(_rpn_box_loss(
                box_outputs=box_outputs[level],
                box_targets=box_targets_at_level,
                normalizer=1.0
            ))

        # Sum per level losses to total loss.
        rpn_score_loss = tf.add_n(score_losses)
        rpn_box_loss = tf.add_n(box_losses)

        return rpn_score_loss, rpn_box_loss

