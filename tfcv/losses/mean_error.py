import tensorflow as tf

def mean_absolute_error(y_true, y_pred, name_scope='MAE'):
    with tf.name_scope(name_scope):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)

        return tf.math.reduce_mean(tf.math.abs(y_pred - y_true), axis=-1)

def mean_squared_error(y_true, y_pred, name_scope='MSE'):
    with tf.name_scope(name_scope):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)

        return tf.math.reduce_mean(tf.math.squared_difference(y_pred - y_true), axis=-1)
