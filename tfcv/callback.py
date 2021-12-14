import tensorflow as tf
from sdk.core.trainer import Trainer

__all__ = ['CallbackList', 'Callback']

class CallbackList(object):

    def __init__(self, callbacks, trainer=None):
        self.callbacks=tf.nest.flatten(callbacks) if callbacks else []
        if trainer:
            self.set_trainer(trainer)

    def set_trainer(self, trainer: Trainer):
        self.trainer=trainer
        for cb in self.callbacks:
            cb.set_trainer(trainer)

    def before_train(self, train_steps):
        for cb in self.callbacks:
            cb.before_train(train_steps)
            
    def before_epoch(self, steps):
        for cb in self.callbacks:
            cb.before_epoch(steps)

    def trigger_epoch(self, train_metrics, eval_metrics):
        add={}
        for cb in self.callbacks:
            msg=cb.trigger_epoch(train_metrics, eval_metrics)
            if msg is not None:
                add.update({cb.name:msg})
        return add

    def after_epoch(self, train_metrics, eval_metrics):
        additional_msg=self.trigger_epoch(train_metrics, eval_metrics)
        for cb in self.callbacks:
            cb.after_epoch(train_metrics, eval_metrics, additional_msg)

    def after_train(self, final_eval_metrics):
        additional_msg=self.trigger_train(final_eval_metrics)
        for cb in self.callbacks:
            cb.after_train(final_eval_metrics, additional_msg)

    def trigger_train(self, final_eval_metrics):
        msg = dict()
        for cb in self.callbacks:
            tmp = cb.trigger_train(final_eval_metrics)
            if tmp is not None:
                msg.update({cb.name:tmp})
        return msg
        
    def trigger_step(self):
        for cb in self.callbacks:
            cb.trigger_step()
    
    def after_run(self):
        for cb in self.callbacks:
            cb.after_run()

class Callback(object):
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

                

