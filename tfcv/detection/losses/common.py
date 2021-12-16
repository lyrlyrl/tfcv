import tensorflow as tf

def _huber_loss(y_true, y_pred, weights, delta):

    num_non_zeros = tf.math.count_nonzero(weights, dtype=tf.float32)

    huber_keras_loss = tf.keras.losses.Huber(
        delta=delta,
        reduction=tf.keras.losses.Reduction.SUM,
        name='huber_loss'
    )

    y_true = tf.expand_dims(y_true, axis=-1)
    y_pred = tf.expand_dims(y_pred, axis=-1)

    huber_loss = huber_keras_loss(
        y_true,
        y_pred,
        sample_weight=weights
    )

    assert huber_loss.dtype == tf.float32

    huber_loss = tf.math.divide_no_nan(huber_loss, num_non_zeros, name="huber_loss")

    assert huber_loss.dtype == tf.float32
    return huber_loss


def _sigmoid_cross_entropy(multi_class_labels, logits, weights, sum_by_non_zeros_weights=False):

    assert weights.dtype == tf.float32

    sigmoid_cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=multi_class_labels,
        logits=logits,
        name="x-entropy"
    )

    assert sigmoid_cross_entropy.dtype == tf.float32

    sigmoid_cross_entropy = tf.math.multiply(sigmoid_cross_entropy, weights)
    sigmoid_cross_entropy = tf.math.reduce_sum(input_tensor=sigmoid_cross_entropy)

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


def _softmax_cross_entropy(onehot_labels, logits):

    num_non_zeros = tf.math.count_nonzero(onehot_labels, dtype=tf.float32)

    softmax_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        labels=tf.stop_gradient(onehot_labels),
        logits=logits
    )

    assert softmax_cross_entropy.dtype == tf.float32

    softmax_cross_entropy = tf.math.reduce_sum(input_tensor=softmax_cross_entropy)
    softmax_cross_entropy = tf.math.divide_no_nan(
        softmax_cross_entropy,
        num_non_zeros,
        name="softmax_cross_entropy"
    )

    assert softmax_cross_entropy.dtype == tf.float32
    return softmax_cross_entropy

