import tensorflow as tf

__all__ = ['get_strategy']

def get_strategy(cfg):
    if cfg.num_gpus > 1:
        strategy = tf.distribute.MirroredStrategy(
            devices=["device:GPU:%d" % i for i in range(cfg.num_gpus)]
        )
    elif cfg.num_gpus == 1:
        strategy = tf.distribute.OneDeviceStrategy('device:GPU:0')
    else:
        strategy = tf.distribute.OneDeviceStrategy('device:CPU:0')
    cfg.replicas = strategy.num_replicas_in_sync
    return strategy