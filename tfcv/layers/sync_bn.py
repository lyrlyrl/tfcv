import tensorflow as tf

def SyncBatchNormalization(*args, **kwargs):
    return tf.keras.layers.experimental.SyncBatchNormalization(*args, **kwargs)