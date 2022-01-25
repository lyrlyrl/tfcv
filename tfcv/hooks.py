import logging
from typing import Callable
import os

import tensorflow as tf

from tensorflow.keras import Model

from tfcv.distribute import MPI_is_distributed, MPI_local_rank
from tfcv.utils.lazy_import import LazyImport

hvd = LazyImport('horovod.tensorflow')

__all__ = ['HookList', 'Hook', 'CheckpointAndBroadcastHook', 'LoggerHook']

class HookList(object):

    def __init__(self, hooks, trainer=None):
        self._logger = logging.getLogger(name = 'hook')
        self.hooks=tf.nest.flatten(hooks) if hooks else []
        if trainer:
            self.set_trainer(trainer)

    def set_trainer(self, trainer):
        self.trainer=trainer
        for cb in self.hooks:
            cb.set_trainer(trainer)

    def before_train(self):
        for cb in self.hooks:
            cb.before_train()
            
    def before_epoch(self, steps, current_step, epoch_number=None):
        for cb in self.hooks:
            cb.before_epoch(steps, current_step, epoch_number)

    def after_epoch(self, train_throuput, train_loss, metrics):
        self._logger.info(f'train_throuput: {train_throuput}')

        self._logger.info(f'train_loss: {train_loss}')

        for name, value in metrics.items():
            self._logger.info(f'{name}: {value}')

        # additional_msg=self.trigger_epoch(train_metrics, eval_metrics)
        for cb in self.hooks:
            cb.after_epoch(train_throuput, train_loss, metrics)

    def after_train(self, success, addtional_message=None):
        for cb in self.hooks:
            cb.after_train(success, addtional_message)
        
    def after_train_batch(self, outputs):
        for cb in self.hooks:
            cb.after_train_batch(outputs)
    
    def after_run(self):
        for cb in self.hooks:
            cb.after_run()
    
    def after_evaluate(self, outputs):
        for cb in self.hooks:
            cb.after_evaluate(outputs)

class Hook(object):
    # trackable = []
    def __init__(self, name=None):
        if not name:
            name = 'DefaultHook'
        self._name = name
        self.logger = logging.getLogger(name = self.name)
    @property
    def name(self):
        return self._name
    @property
    def chief_only(self):
        return self._chief_only
    def set_trainer(self, trainer):
        self.trainer=trainer
    def before_train(self):
        pass
    def before_epoch(self, steps, current_step, epoch_number):
        pass
    def after_epoch(self, train_throuput, train_loss, metrics):
        pass
    def after_train(self, success, addtional_message):
        pass
    def after_train_batch(self, outputs):
        pass
    def after_run(self):
        pass
    def after_evaluate(self, outputs):
        pass

class CheckpointAndBroadcastHook(Hook):

    def __init__(
            self,
            checkpoint: tf.train.Checkpoint,
            ckpt_dir,
            ckpt_name,
            ckpt_interval=10000,
            force_sync=False,
            initialize_fn: Callable[[Model], bool] = None):
        super().__init__(name='checkpoint')
        self._ckpt_dir = ckpt_dir
        self._ckpt_name = ckpt_name
        self._ckpt_interval = ckpt_interval
        self._checkpoint = checkpoint
        self._latest_step = 0
        self._initialize_fn = initialize_fn
        self._force_sync = force_sync
        self._logger = logging.getLogger(self.name)
    def set_trainer(self, trainer):
        super(CheckpointAndBroadcastHook, self).set_trainer(trainer)
    def save(self):
        if MPI_local_rank() == 0:
            self._checkpoint.save(os.path.join(self._ckpt_dir, self._ckpt_name))
    def broadcast(self):
        hvd.broadcast_variables(self.trainer._model.variables, root_rank=0)
        hvd.broadcast_variables(self.trainer._optimizer.variables(), root_rank=0)
        hvd.broadcast_variables([self.trainer._global_step], root_rank=0)
    def before_train(self):
        if MPI_local_rank() == 0:
            latest = tf.train.latest_checkpoint(self._ckpt_dir)
            if latest != None:
                self._checkpoint.restore(latest)
                self._logger.info(f'restored from checkpoint {latest}')
            elif self._initialize_fn != None:
                assert self._initialize_fn(self.trainer.model)
                self._logger.info('initialized with function')
            else:
                self._logger.info('random initialized')
        if MPI_is_distributed():
            self.broadcast()
    def before_epoch(self, steps, *args):
        self._latest_step += steps
    def after_epoch(self, *args):
        if self._latest_step >= self._ckpt_interval:
            self.save()
            self._latest_step = 0
        if self._force_sync and MPI_is_distributed():
            self.broadcast()
    def after_train(self, success, *args):
        if self._latest_step > 0 and success:
            self.save()
    
class LoggerHook(Hook):
    def __init__(self, logger, add_sys_perf=False):
        super().__init__(name='logger')
        self._logger = logger
        self._latest_step = 0
    def before_epoch(self, steps, current_step, epoch_number):
        if epoch_number != None:
            step = (epoch_number, current_step)
        else:
            step = current_step
        self._logger.message(step, f'train loop started to train {steps} steps')
        self._latest_step = current_step + steps
    def after_epoch(self, train_throuput, train_loss, metrics):
        self._logger.perf(self._latest_step, {'train_throuput': train_throuput})
        metrics['train_loss'] = train_loss
        learning_rate = self.trainer.optimizer.learning_rate
        if callable(learning_rate):
            learning_rate = learning_rate()
        try:
            learning_rate = learning_rate.numpy()
        except:
            pass
        metrics['learning_rate'] = learning_rate
        self._logger.metric(self._latest_step, metrics)
    def after_train(self, success, additional_msg = None):
        if success:
            self._logger.finalize(self._latest_step, 'success')
        else:
            assert additional_msg != None
            self._logger.finalize(self._latest_step, additional_msg)