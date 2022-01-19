import logging
import os

import tensorflow as tf
import numpy as np
import yaml

from tfcv.exception import NanTrainLoss

__all__ = ['HookList', 'Hook', 'CheckpointHook', 'LoggerHook']

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

    def before_train(self, train_steps):
        for cb in self.hooks:
            cb.before_train(train_steps)
            
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

    def after_train(self, final_eval_metrics):
        additional_msg=self.trigger_train(final_eval_metrics)
        for cb in self.hooks:
            cb.after_train(final_eval_metrics, additional_msg)
        
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
    def before_train(self, train_steps):
        pass
    def before_epoch(self, steps, current_step, epoch_number):
        pass
    def after_epoch(self, train_throuput, train_loss, metrics):
        pass
    def before_train(self, steps):
        pass
    def after_train(self, final_eval_metrics, additional_msg):
        pass
    def after_train_batch(self, outputs):
        pass
    def after_run(self):
        pass
    def after_evaluate(self, outputs):
        pass

class CheckpointHook(Hook):

    def __init__(
            self,
            checkpoint: tf.train.Checkpoint,
            model_dir,
            ckpt_subdir,
            ckpt_name='ckpt',
            best_metric=None,
            best_subdir='best',
            initialize_fn=None):
        super().__init__(name='best_model')
        self._trained = False
        self._best_metric = best_metric
        self._ckpt_path = os.path.join(model_dir, ckpt_subdir, ckpt_name)
        self._best_dir = os.path.join(model_dir, best_subdir)
        self._best_result_dir = os.path.join(self._best_dir, 'best.yaml')
        if not os.path.isdir(self._best_dir):
            os.makedirs(self._best_dir)
            self.best_results={}
        else:
            if os.path.isfile(self._best_result_dir):
                with open(self._best_result_dir, 'r') as fp:
                    self.best_results = yaml.load(fp, Loader=yaml.CLoader)
        self._initialize_fn = initialize_fn

    def set_trainer(self, trainer):
        super().set_trainer(trainer)
        self._checkpoint = tf.train.Checkpoint(model=self.trainer.model, optimizer=self.trainer.optimizer)

    def before_train(self, *args, **kwargs):
        self._trained = True

    def after_evaluate(self, outputs):
        if self._trained:
            pass
    
class LoggerHook(Hook):
    def __init__(self, logger):
        super().__init__(name='logger')
        self._logger = logger
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
        metrics['learning_rate'] = learning_rate.numpy()
        self._logger.metric(self._latest_step, metrics)
    def after_train(self, success, additional_msg = None):
        if success:
            self._logger.finalize(self._latest_step, True)
        else:
            assert additional_msg != None
            self._logger.finalize(self._latest_step, additional_msg)