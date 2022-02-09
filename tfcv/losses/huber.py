import tensorflow as tf

def huber_loss(*args, **kwargs):
    return tf.keras.losses.huber(*args, **kwargs)