import tensorflow as tf

def SyncBatchNormalization(*args, runtime='tf', **kwargs):
    if runtime == 'tf':
        return tf.keras.layers.experimental.SyncBatchNormalization(*args, **kwargs)