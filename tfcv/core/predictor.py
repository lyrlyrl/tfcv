from typing import List

import tensorflow as tf

from tfcv.utils.lazy_import import LazyImport
hvd = LazyImport('horovod.tensorflow')

__all__ = ['Predictor', 'merge_replica_results', 'HorovodPredictor']

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

    @property
    def model(self):
        return self._model

    @tf.function
    def predict_step(self, inputs):
        return self._task.inference_forward(self._model, inputs)
    
    @tf.function
    def service_step(self, image):
        inputs = self._task.preprocess(image)
        outputs = self._task.inference_forward(self._model, inputs)
        return self._task.postprocess(outputs)

class HorovodPredictor(Predictor):
    def __init__(self, *args, **kwargs):
        super(HorovodPredictor, self).__init__(*args, **kwargs)
        assert hvd.is_initialized()

    @tf.function
    def predict_step(self, inputs):
        replica_outputs = self._task.inference_forward(self._model, inputs)
        all_outputs = tf.nest.map_structure(hvd.allgather, replica_outputs)
        return all_outputs

    @tf.function
    def service_step(self, image):
        inputs = self._task.preprocess(image)
        outputs = self._task.inference_forward(self._model, inputs)
        replica_outputs = self._task.postprocess(outputs)
        all_outputs = tf.nest.map_structure(hvd.allgather, replica_outputs)
        return all_outputs