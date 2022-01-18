import math
import abc
import tensorflow as tf
import numpy as np

import time

from tfcv import logger
from tfcv.distribute import MPI_is_distributed, MPI_size
from tfcv.exception import NanTrainLoss
from tfcv.utils.progress import get_tqdm

__all__ = ['Trainer']

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
    def train_loss(self):
        """Accesses the training loss metric object."""
        return self._train_loss

    def compile(self):
        strategy = tf.distribute.get_strategy()
        
        train_step_fn = tf.function(self.train_step)
        def dist_train_step(iterator):
            strategy.run(
                train_step_fn,
                args=(next(iterator),)
            )
        self._train_op = tf.function(dist_train_step)

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
                outputs = self._train_op(train_iterator)
                self.hooks.after_train_batch(outputs)

            train_throuput, train_loss, metrics = self.train_loop_end()
            
            if np.isnan(train_loss):
                raise NanTrainLoss(current_step, metrics)

            self.hooks.after_epoch(train_throuput, train_loss, metrics)
            current_step += steps_to_run
            metrics['train_loss'] = train_loss
            logger.metric(current_step+start_step, metrics)
            logger.perf(current_step+start_step, {'train_throuput': train_throuput})

    def train_forward(self, inputs):
        model_outputs = self._model(inputs, training=True)
        raw_loss = tf.math.reduce_sum(self._model.losses)
        return (raw_loss, None, model_outputs) # total_loss, to_update, to_output

    def train_step(self, inputs):
        with tf.GradientTape() as tape:
            raw_loss, to_update, to_output = self.train_forward(inputs)
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
        if MPI_is_distributed():
            throuput *= MPI_size()
        train_loss = self._train_loss.result().numpy()
        
        metrics = {metric.name: metric.result().numpy() for metric in self._metrics}
        metrics.update({metric.name: metric.result().numpy() for metric in self._model._metrics})

        return throuput, train_loss, metrics