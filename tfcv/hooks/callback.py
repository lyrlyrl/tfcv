import tensorflow as tf
from tfcv.core.trainer import Trainer

__all__ = ['HookList', 'Hook']

class HookList(object):

    def __init__(self, hooks, trainer=None):
        self.hooks=tf.nest.flatten(hooks) if hooks else []
        if trainer:
            self.set_trainer(trainer)

    def set_trainer(self, trainer: Trainer):
        self.trainer=trainer
        for cb in self.hooks:
            cb.set_trainer(trainer)

    def before_train(self, train_steps):
        for cb in self.hooks:
            cb.before_train(train_steps)
            
    def before_epoch(self, steps):
        for cb in self.hooks:
            cb.before_epoch(steps)

    def trigger_epoch(self, train_metrics, eval_metrics):
        add={}
        for cb in self.hooks:
            msg=cb.trigger_epoch(train_metrics, eval_metrics)
            if msg is not None:
                add.update({cb.name:msg})
        return add

    def after_epoch(self, train_metrics, eval_metrics):
        additional_msg=self.trigger_epoch(train_metrics, eval_metrics)
        for cb in self.hooks:
            cb.after_epoch(train_metrics, eval_metrics, additional_msg)

    def after_train(self, final_eval_metrics):
        additional_msg=self.trigger_train(final_eval_metrics)
        for cb in self.hooks:
            cb.after_train(final_eval_metrics, additional_msg)

    def trigger_train(self, final_eval_metrics):
        msg = dict()
        for cb in self.hooks:
            tmp = cb.trigger_train(final_eval_metrics)
            if tmp is not None:
                msg.update({cb.name:tmp})
        return msg
        
    def trigger_step(self):
        for cb in self.hooks:
            cb.trigger_step()
    
    def after_run(self):
        for cb in self.hooks:
            cb.after_run()

class Hook(object):
    # trackable = []
    def __init__(self, config):
        self._name=config.name
        self._chief_only=config.chief_only
    @property
    def name(self):
        return self._name
    @property
    def chief_only(self):
        return self._chief_only
    def set_trainer(self, trainer: Trainer):
        self.trainer=trainer
    def before_train(self, train_steps):
        pass
    def before_epoch(self, steps):
        pass
    def trigger_epoch(self, train_metrics, eval_metrics):
        pass
    def after_epoch(self, train_metrics, eval_metrics, add_msg):
        pass
    def before_train(self, steps):
        pass
    def after_train(self, final_eval_metrics, additional_msg):
        pass
    def trigger_train(self, final_eval_metrics):
        pass
    def trigger_step(self):
        pass
    def after_run(self):
        pass

                

