import math
import abc
import tensorflow as tf
import numpy as np

import time

from tfcv import logger
from tfcv.distribute import MPI_is_distributed, MPI_size
from tfcv.utils.lazy_import import LazyImport
from tfcv.exception import NanTrainLoss
from tfcv.utils.progress import get_tqdm

hvd = LazyImport('horovod.tensorflow')

__all__ = ['DefaultTrainer']

class Trainer(tf.Module, metaclass=abc.ABCMeta):
    def __init__(
            self, 
            params,
            global_step: tf.Variable,
            model: tf.keras.Model,
            task,
            optimizer=None,
            metrics=[],
            hooks=None):
        super(Trainer, self).__init__(name='trainer')
        self._params = params
        self._global_step = global_step
        self._model = model
        self._task = task
        self._optimizer = optimizer
        self._metrics = metrics

        self._train_loss = tf.keras.metrics.Mean(name='train_loss')

        self._train_op = None
        self._validation_op = None
        
        self._train_timer = 0

        self.hooks = hooks
        if self.hooks:
            self.hooks.set_trainer(self)

    @property
    def model(self):
        return self._model

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def task(self):
        return self._task

    @property
    def train_loss(self):
        """Accesses the training loss metric object."""
        return self._train_loss

    def train(self, num_steps, train_iterator):
        if not self._train_op:
            raise 
        assert num_steps > 0
        num_loops = math.ceil(num_steps / self._params.solver.steps_per_loop)
        start_step = self._global_step.numpy()
        current_step = 0
        for loop_number in range(num_loops):
            steps_to_run = (loop_number+1) * self._params.solver.steps_per_loop - current_step
            self.hooks.before_epoch(steps_to_run, current_step + start_step)            
            self.train_loop_begin()
                
            for _ in get_tqdm(
                    range(steps_to_run),
                    desc=f'start at step {current_step + start_step}: '):
                outputs = self._train_op(next(train_iterator))
                self.hooks.after_train_batch(outputs)

            train_throuput, train_loss, metrics = self.train_loop_end()
            
            if np.isnan(train_loss):
                raise NanTrainLoss(current_step, metrics)

            self.hooks.after_epoch(train_throuput, train_loss, metrics)
            current_step += steps_to_run

    @abc.abstractmethod
    def compile(self):
        pass

    def train_loop_begin(self):
        self._train_loss.reset_state()

        for metric in self._metrics:
            metric.reset_state()
        
        for metric in self._model.metrics:
            metric.reset_state()

        self._train_timer = time.time()
    
    def train_loop_end(self):
        times = time.time() - self._train_timer

        throuput = self._params.global_train_batch_size * self._params.solver.steps_per_loop / times
        train_loss = self._train_loss.result().numpy()
        
        metrics = {metric.name: metric.result().numpy() for metric in self._metrics}
        metrics.update({metric.name: metric.result().numpy() for metric in self._model._metrics})

        return throuput, train_loss, metrics

class DefaultTrainer(Trainer):

    def compile(self):
        strategy = tf.distribute.get_strategy()
        
        def dist_train_step(inputs):
            outputs = strategy.run(
                self.train_step,
                args=(inputs,)
            )
            return outputs

        self._train_op = tf.function(dist_train_step)

    def train_step(self, inputs):
        with tf.GradientTape() as tape:
            raw_loss, to_update, to_output = self._task.train_forward(self._model, inputs)
            if isinstance(self._optimizer, tf.keras.mixed_precision.LossScaleOptimizer):
                loss = self._optimizer.get_scaled_loss(raw_loss)
            else:
                loss = raw_loss
        trainable_weights = self._model.trainable_weights
        grads = tape.gradient(loss, trainable_weights)
        if isinstance(self._optimizer, tf.keras.mixed_precision.LossScaleOptimizer):
            grads = self._optimizer.get_unscaled_gradients(grads)
        self._optimizer.apply_gradients(list(zip(grads, trainable_weights)))
        self._train_loss.update_state(raw_loss)
        for metric in self._metrics:
            metric.update_state(to_update[metric.name])
        self._global_step.assign_add(1)
        return to_output

class HorovodTrainer(Trainer):
    
    def __init__(self, *args, **kwargs):
        super(HorovodTrainer, self).__init__(*args, **kwargs)
        assert MPI_is_distributed()
        assert hvd.is_initialized()
    
    def compile(self):
        def dist_train_op(inputs, first_batch=False):
            outputs = self.train_step(inputs, first_batch)
            return outputs
        self._train_op = tf.function(dist_train_op)

    def train_loop_end(self):
        throuput, train_loss, metrics = super(HorovodTrainer, self).train_loop_end()
        throuput *= MPI_size()
        return throuput, train_loss, metrics
    
    def train_step(self, inputs, first_batch):
        with tf.GradientTape() as tape:
            raw_loss, to_update, to_output = self._task.train_forward(self._model, inputs)
            if isinstance(self._optimizer, tf.keras.mixed_precision.LossScaleOptimizer):
                loss = self._optimizer.get_scaled_loss(raw_loss)
            else:
                loss = raw_loss

        tape = hvd.DistributedGradientTape(tape)

        trainable_weights = self._model.trainable_weights
        grads = tape.gradient(loss, trainable_weights)
        if isinstance(self._optimizer, tf.keras.mixed_precision.LossScaleOptimizer):
            grads = self._optimizer.get_unscaled_gradients(grads)
        self._optimizer.apply_gradients(list(zip(grads, trainable_weights)))
        if first_batch:
            hvd.broadcast_variables(self._model.variables, root_rank=0)
            hvd.broadcast_variables(self._optimizer.variables(), root_rank=0)
            hvd.broadcast_variables(self._global_step, root_rank=0)
        else:
            self._train_loss.update_state(raw_loss)
            for metric in self._metrics:
                metric.update_state(to_update[metric.name])
            self._global_step.assign_add(1)
        return to_output

    def train(self, num_steps, train_iterator):
        if not self._train_op:
            raise 
        assert num_steps > 0
        # initlize variables across all ranks
        self._train_op(next(train_iterator), True)

        num_loops = math.ceil(num_steps / self._params.solver.steps_per_loop)
        start_step = self._global_step.numpy()
        current_step = 0
        for loop_number in range(num_loops):
            steps_to_run = (loop_number+1) * self._params.solver.steps_per_loop - current_step
            self.hooks.before_epoch(steps_to_run, current_step + start_step)            
            self.train_loop_begin()
                
            for _ in get_tqdm(
                    range(steps_to_run),
                    desc=f'start at step {current_step + start_step}: '):
                outputs = self._train_op(next(train_iterator))
                self.hooks.after_train_batch(outputs)

            train_throuput, train_loss, metrics = self.train_loop_end()
            
            if np.isnan(train_loss):
                raise NanTrainLoss(current_step, metrics)

            self.hooks.after_epoch(train_throuput, train_loss, metrics)
            current_step += steps_to_run