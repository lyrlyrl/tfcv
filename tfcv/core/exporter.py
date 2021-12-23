import abc
import tensorflow as tf

__all__ = ['Exporter']

class Exporter(tf.Module, metaclass=abc.ABCMeta):

    def __init__(self, model, params, name=None):
        super().__init__(name=name)
        self._model = model
        self._params = params

    @abc.abstractmethod
    def inference_step(self, image):
        pass
    
    @tf.function
    def inference_from_tensor(self, images):
        return self.inference_step(images)