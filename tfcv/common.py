import os
import logging

import tensorflow as tf
import numpy as np

__all__ = ['set_xla', 'set_amp', 'create_global_step', 'autocast']

def set_xla(cfg):
    if cfg.xla:
        # os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit'
        tf.config.optimizer.set_jit('autoclustering')
        logging.info('XLA is activated')

def set_amp(cfg):
    if cfg.amp:
        policy = tf.keras.mixed_precision.experimental.Policy("mixed_float16")
        tf.keras.mixed_precision.experimental.set_policy(policy)
        logging.info('AMP is activated')

def create_global_step(name="global_step") -> tf.Variable:
    """Creates a `tf.Variable` suitable for use as a global step counter.

    Creating and managing a global step variable may be necessary for
    `AbstractTrainer` subclasses that perform multiple parameter updates per
    `Controller` "step", or use different optimizers on different steps.

    In these cases, an `optimizer.iterations` property generally can't be used
    directly, since it would correspond to parameter updates instead of iterations
    in the `Controller`'s training loop. Such use cases should simply call
    `step.assign_add(1)` at the end of each step.

    Returns:
        A non-trainable scalar `tf.Variable` of dtype `tf.int64`, with only the
        first replica's value retained when synchronizing across replicas in
        a distributed setting.
    """
    return tf.Variable(
        0,
        dtype=tf.int64,
        name=name,
        trainable=False,
        aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)

def autocast(data):
    if isinstance(data, (np.ndarray, np.generic)):
        data = data.tolist()
    elif isinstance(data, tf.Tensor):
        data = data.numpy().tolist()
    return data