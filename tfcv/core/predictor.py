import time
import abc
from typing import List
import functools

import tensorflow as tf
from tfcv import logger
from tfcv.utils.progress import get_tqdm
from tfcv.hooks import HookList, Hook

__all__ = ['Predictor', 'merge_replica_results']

def merge_replica_results(strategy, inputs):
    dist_values = strategy.experimental_local_results(inputs)
    def _merge(*args):
        return tf.concat(args, 0)
    return tf.nest.map_structure(_merge, *dist_values)

class Predictor(tf.Module):
    def __init__(
            self, 
            params,
            model,
            task):
        super(Predictor, self).__init__(name='predictor')
        self._params = params
        self._model = model
        self._task = task
        self._predict_op = None
        self._strategy = None

    @property
    def model(self):
        return self._model

    def compile(self):
        strategy = tf.distribute.get_strategy()
        def dist_predict_step(inputs):
            outputs = strategy.run(
                self.predict_step,
                args=(inputs,)
            )
            return outputs
        if strategy != None:
            self._predict_op = tf.function(dist_predict_step)
        else:
            self._predict_op = tf.function(self.predict_step)
        self._strategy = strategy
    def predict_step(self, inputs):
        return self._task.inference_forward(self._model, inputs)
    def predict_batch(self, inputs):
        outputs = self._predict_op(inputs)
        if self._strategy:
            outputs = merge_replica_results(self._strategy, outputs)
        return outputs