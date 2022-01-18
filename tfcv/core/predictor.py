import time
import abc
import tensorflow as tf
from tfcv import logger

__all__ = ['Predictor', 'merge_replica_results']

def merge_replica_results(strategy, inputs):
    dist_values = strategy.experimental_local_results(inputs)
    def _merge(*args):
        return tf.concat(args, 0)
    return tf.nest.map_structure(_merge, *dist_values)

class Predictor(tf.Module, metaclass=abc.ABCMeta):
    def __init__(
            self, 
            params,
            model,
            task,
            hooks=None):
        super(Predictor, self).__init__(name='predictor')
        self._params = params
        self._model = model
        self._task = task
    @property
    def model(self):
        return self._model
    def predict(self, inputs):
        return self.inference_step(inputs)
    def inference_step(self, inputs):
        return self._model(inputs, training=False)
    