import tensorflow as tf

def sigmoid_crossentropy(multi_class_labels, logits, weights, sum_by_non_zeros_weights=False):

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


def softmax_crossentropy(onehot_labels, logits):

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