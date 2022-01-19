import abc
import tensorflow as tf

__all__ = ['Task']

class Task(object, metaclass = abc.ABCMeta):
    @abc.abstractmethod
    def create_model(self):
        pass
    @abc.abstractmethod
    def train_forward(self, model, inputs):
        model_outputs = model(inputs, training=True)
        raw_loss = tf.math.reduce_sum(model.losses)
        return (raw_loss, None, model_outputs) # total_loss, to_update, to_output
    @abc.abstractmethod
    def inference_forward(self, model, inputs):
        pass