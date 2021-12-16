import abc

__all__ = ['PyMetric']

# replacement of tf.keras.metrics.Metric, used in evaluate process
class PyMetric(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def reset_state(self):
        pass
    @abc.abstractmethod
    def result(self):
        pass
    @abc.abstractmethod
    def update_state(self, groundtruths, predictions):
        pass