import math
import abc
from typing import List
import tensorflow as tf
import numpy as np

import time

from tfcv.hooks import HookList, Hook
from tfcv.distribute import MPI_is_distributed, MPI_size
from tfcv.utils.lazy_import import LazyImport
from tfcv.exception import NanTrainLoss, ManuallyInterrupt

hvd = LazyImport('horovod.tensorflow')

__all__ = ['HorovodTrainer', 'DefaultTrainer']

class Fatal:
    NAN_LOSS = 'train loss nan'
    DATA_EXAUSTED = 'not enough data'

class DefaultTrainer(tf.Module):
    def __init__(
            self, 
            params,
            global_step: tf.Variable,
            model: tf.keras.Model,
            task,
            optimizer=None,
            metrics=[],
            hooks: List[Hook] = []):
        super(DefaultTrainer, self).__init__(name='trainer')
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

        self.hooks = HookList(hooks, self)

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
    def global_step(self):
        return self._global_step

    @property
    def train_loss(self):
        """Accesses the training loss metric object."""
        return self._train_loss

    def train(self, num_steps, train_iterator):
        try:
            self.compile()
            # warm step
            self._train_op(next(train_iterator))
            self.hooks.before_train()

            num_steps = num_steps - self._global_step.numpy()
            assert num_steps > 0

            num_loops = math.ceil(num_steps / self._params.solver.steps_per_loop)
            start_step = self._global_step.numpy()
            current_step = 0
            for loop_number in range(num_loops):
                steps_to_run = (loop_number+1) * self._params.solver.steps_per_loop - current_step
                self.hooks.before_epoch(steps_to_run, current_step + start_step)            
                self.train_loop_begin()
                for _ in range(steps_to_run):
                    outputs = self._train_op(next(train_iterator))
                    self.hooks.after_train_batch(outputs)

                train_throuput, train_loss, metrics = self.train_loop_end()
                
                if np.isnan(train_loss):
                    raise NanTrainLoss(current_step, metrics)

                self.hooks.after_epoch(train_throuput, train_loss, metrics)
                current_step += steps_to_run
        except NanTrainLoss:
            self.hooks.after_train(False, Fatal.NAN_LOSS)
            code = Fatal.NAN_LOSS
        except tf.errors.OutOfRangeError:
            self.hooks.after_train(False, Fatal.DATA_EXAUSTED)
            code = Fatal.DATA_EXAUSTED
        except (KeyboardInterrupt, ManuallyInterrupt):
            self.hooks.after_train(False, 'manually')
            code = 'manually'
        except Exception as e:
            self.hooks.after_train(False, e)
            code = str(e)
        else:
            self.hooks.after_train(True)
            code = 0
        finally:
            self.hooks.after_run()
            return code

    def compile(self, force=False):
        if force or self._train_op == None:
            self._train_op = tf.function(self.train_step)

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
        # update
        self._train_loss.update_state(raw_loss)
        for metric in self._metrics:
            metric.update_state(*to_update[metric.name])
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
        train_loss = self._train_loss.result().numpy()
        
        metrics = {metric.name: metric.result().numpy() for metric in self._metrics}
        metrics.update({metric.name: metric.result().numpy() for metric in self._model._metrics})

        return throuput, train_loss, metrics

class HorovodTrainer(DefaultTrainer):
    
    def __init__(self, *args, **kwargs):
        super(HorovodTrainer, self).__init__(*args, **kwargs)
        assert hvd.is_initialized()
    
    def train_step(self, inputs):
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
        # update
        self._train_loss.update_state(raw_loss)
        for metric in self._metrics:
            metric.update_state(*to_update[metric.name])
        self._global_step.assign_add(1)
        return to_output
