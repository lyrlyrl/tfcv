import math
from typing import Optional
import dataclasses
import tensorflow as tf
from absl import logging
import abc

__all__ = ['StopTraining', 'Trainer']


class StopTraining(Exception):
    """
    An exception thrown to stop training.
    """
    pass

class DummyContextManager(object):

    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass

class Trainer(tf.Module, metaclass=abc.ABCMeta):
    """
    Whether this process is the chief worker in distributed training.
    Certain callbacks will only be run by chief worker.
    """
    def __init__(self,
                task,
                model,
                total_steps,
                steps_per_loop,
                steps_per_evaluate,
                strategy,
                train_dataset=None,
                val_dataset=None,
                optimizer=None,
                is_chief=True):
        self._task = task
        self._model = model
        self._total_steps = total_steps
        self._steps_per_loop= steps_per_loop
        self._steps_per_evaluate = steps_per_evaluate
        self._train_dataset = train_dataset
        self._val_dataset = val_dataset
        self._optimizer = optimizer

        self._global_step = self.create_global_step()
        self._checkpoint = None
        self._is_chief = is_chief
        # self._eager = False

        self._compiled = False
        self._strategy = strategy

        
    def create_global_step(self):
        return tf.Variable(
                0,
                dtype=tf.int64,
                name="global_step",
                trainable=False,
                aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)

    @property
    def task(self):
        return self._task

    @property
    def model(self):
        return self._model

    @property
    def compiled(self):
        return self._compiled

    @compiled.setter
    def compiled(self, value):
        self._compiled = value
    
    @property
    def strategy(self):
        return self._strategy
    
    @property
    def strategy_scope(self):
        if self.strategy is not None:
            strategy_scope = self.strategy.scope()
        else:
            strategy_scope = DummyContextManager()
        return strategy_scope

    @property
    def checkpoint(self):
        return self._checkpoint

    @checkpoint.setter
    def checkpoint(self, value):
        self._checkpoint = value

    @property
    def is_chief(self):
        return self._is_chief
    
    # @is_chief.setter
    # def is_chief(self, value):
    #     self._is_chief = value
        
    @property
    def optimizer(self):
        return self._optimizer
    
    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value

    @property
    def global_step(self):
        return self._global_step

    @property
    def train_dataset(self):
        return self._train_dataset
    
    @property
    def val_dataset(self):
        return self._val_dataset

    @property
    def total_steps(self):
        return self._total_steps
    
    @property
    def steps_per_loop(self):
        return self._steps_per_loop

    @property
    def steps_per_evaluate(self):
        return self._steps_per_evaluate

    @classmethod
    def initialize(cls):
        logging.info('trainer initialized')
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    
    @classmethod
    def compute_total_steps(cls, num_examples, global_batch_size, num_epochs):
        steps_per_epoch = math.ceil(num_examples / global_batch_size)
        total_steps = num_epochs * steps_per_epoch
        return total_steps
    
    @abc.abstractmethod
    def make_train_op(self):
        return

    def initialize_model(self):
        self.task.initialize(self.model)
    
    def trainer_callback(self):
        return None
