from tfcv.hooks.base import Hook

class TestHook(Hook):
    def before_train(self, train_steps):
        self.logger.info('test before train')
    def before_epoch(self, steps):
        self.logger.info('test before epoch')
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
    def after_train_batch(self, outputs):
        pass
    def after_run(self):
        pass