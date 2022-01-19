import tensorflow as tf
from tfcv.distribute import MPI_is_distributed
from tfcv.utils.lazy_import import LazyImport

hvd = LazyImport('horovod.tensorflow')

def BatchNormalization(*args, **kwargs):
    strategy = tf.distribute.get_strategy()
    trainable = kwargs.get('trainable', True)
    if strategy != None and strategy.num_replicas_in_sync > 1 and trainable:
        return tf.keras.layers.experimental.SyncBatchNormalization(*args, **kwargs)
    elif MPI_is_distributed() and trainable:
        return hvd.SyncBatchNormalization(*args, **kwargs)
    else:
        return tf.keras.layers.BatchNormalization(*args, **kwargs)