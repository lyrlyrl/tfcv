import tensorflow as tf

def focal_loss(y_true, y_pred, alpha, gamma):
    with tf.name_scope('focal_loss'):
        y_true = tf.cast(y_true, dtype=tf.float32)
        y_pred = tf.cast(y_pred, dtype=tf.float32)
        positive_label_mask = tf.equal(y_true, 1.0)
        cross_entropy = (
            tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred))
        probs = tf.sigmoid(y_pred)
        probs_gt = tf.where(positive_label_mask, probs, 1.0 - probs)
        # With small gamma, the implementation could produce NaN during back prop.
        modulator = tf.pow(1.0 - probs_gt, gamma)
        loss = modulator * cross_entropy
        weighted_loss = tf.where(positive_label_mask, alpha * loss,
                                (1.0 - alpha) * loss)
    return weighted_loss