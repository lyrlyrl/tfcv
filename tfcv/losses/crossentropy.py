import tensorflow as tf

def sigmoid_crossentropy(multi_class_labels, logits, weights, sum_by_non_zeros_weights=False):

    assert weights.dtype == tf.float32

    sigmoid_cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=multi_class_labels,
        logits=logits,
        name="x-entropy"
    )

    sigmoid_cross_entropy = tf.math.multiply(sigmoid_cross_entropy, weights)
    sigmoid_cross_entropy = tf.math.reduce_sum(sigmoid_cross_entropy)

    assert sigmoid_cross_entropy.dtype == tf.float32

    if sum_by_non_zeros_weights:
        num_non_zeros = tf.math.count_nonzero(weights, dtype=tf.float32)
        sigmoid_cross_entropy = tf.math.divide_no_nan(
            sigmoid_cross_entropy,
            num_non_zeros,
            name="sum_by_non_zeros_weights"
        )

    assert sigmoid_cross_entropy.dtype == tf.float32
    return sigmoid_cross_entropy


def softmax_crossentropy(onehot_labels, logits):

    num_non_zeros = tf.math.count_nonzero(onehot_labels, dtype=tf.float32)

    softmax_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        labels=tf.stop_gradient(onehot_labels),
        logits=logits
    )

    assert softmax_cross_entropy.dtype == tf.float32

    softmax_cross_entropy = tf.math.reduce_sum(softmax_cross_entropy)
    softmax_cross_entropy = tf.math.divide_no_nan(
        softmax_cross_entropy,
        num_non_zeros,
        name="softmax_cross_entropy"
    )

    assert softmax_cross_entropy.dtype == tf.float32
    return softmax_cross_entropy

def categorical_crossentropy(y_true, y_pred, label_smoothing=0):
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    label_smoothing = tf.convert_to_tensor(label_smoothing, y_pred.dtype)

    def _smooth_labels():
        num_classes = tf.cast(tf.shape(y_true)[-1], y_pred.dtype)
        return y_true * (1.0 - label_smoothing) + (label_smoothing / num_classes)

    y_true = tf.cond(label_smoothing, _smooth_labels,
                                    lambda: y_true)
    
    