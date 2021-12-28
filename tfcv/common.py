import os
import logging

import tensorflow as tf

__all__ = ['set_xla', 'set_amp']

def set_xla(cfg):
    if cfg.xla:
        os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit'
        logging.info('XLA is activated')

def set_amp(cfg):
    if cfg.amp:
        policy = tf.keras.mixed_precision.experimental.Policy("mixed_float16", loss_scale="dynamic")
        tf.keras.mixed_precision.experimental.set_policy(policy)
        logging.info('AMP is activated')

