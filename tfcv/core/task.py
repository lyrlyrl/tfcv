import tensorflow as tf
import abc
from typing import Optional

__all__ = ["Task"]

class Task(object, metaclass=abc.ABCMeta):
    """A single-replica view of training procedure.

    Tasks provide artifacts for training/validation procedures, including
    loading/iterating over Datasets, training/validation steps, calculating the
    loss and customized metrics with reduction.
    """
    # Special keys in train/validate step returned logs.
    loss = "loss"
    
    def __init__(self, params, name: Optional[str] = None):
        """Task initialization.

        Args:
            params: the task configuration instance, which can be any of dataclass,
                ConfigDict, namedtuple, etc.
            name: the task name.
        """
        # super().__init__(name=name)
        self._task_config = params
        self._train_num_examples = None
        self._eval_num_examples = None
        self._dtype = 'float32'

    @property
    def task_config(self):
        return self._task_config

    def initialize(self, model: tf.keras.Model):
        pass

    @abc.abstractmethod
    def build_model(self) -> tf.keras.Model:
        """[Optional] Creates model architecture.

        Returns:
            A keras model instance.
        """

    @abc.abstractmethod
    def build_inputs(self, is_training, sample_fn=lambda x: x) -> tf.data.Dataset:
        """Returns a dataset or a nested structure of dataset functions.

        Dataset functions define per-host datasets with the per-replica batch size.
        With distributed training, this method runs on remote hosts.

        Args:
            params: hyperparams to create input pipelines, which can be any of
                dataclass, ConfigDict, namedtuple, etc.
            input_context: optional distribution input pipeline context.

        Returns:
            A nested structure of per-replica input functions.
        """

    @abc.abstractmethod
    def build_evaluate_metric(self):
        return

