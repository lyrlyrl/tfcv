"""Provides a `Controller` class for managing the outer training loop."""

from typing import List
from numpy.lib.function_base import iterable
import tensorflow as tf
import math
import pprint

from absl import logging

from tfcv.core.callback import CallbackList, Callback

from tfcv.core.trainer import *

__all__ = ['Executor']

class Executor:
    
    def __init__(
            self,
            *,    # Makes all args keyword only.
            trainer: Trainer,
            global_step: tf.Variable = None):

        self.trainer = trainer
        self.global_step = global_step or trainer.global_step
        if not isinstance(self.global_step, tf.Variable):
            raise ValueError("`global_step` must be a `tf.Variable`.")

        self.registried_callbacks = []

    def main_loop(self, eager_mode, xla):
        """
        Run the main training loop.

        """
        if xla:
            tf.config.optimizer.set_jit('autoclustering')
        else:
            tf.config.optimizer.set_jit(False)
        try:
            self.registried_callbacks.before_train(self.trainer.total_steps)
            if self.trainer.is_chief:
                inference_op = self.trainer.make_inference_op(eager_mode)
                eval_metrics = self.trainer.task.build_evaluate_metric()
                
            if self.trainer.total_steps > 0:
                # train mode
                assert self.trainer.train_dataset is not None
                # build train dataset iterator
                iterator = iter(self.trainer.train_dataset)
                num_epochs = math.ceil(self.trainer.total_steps / self.trainer.steps_per_loop)

                current_step = self.global_step.numpy()
                last_evaluate_step = current_step
                
                for epoch in range(num_epochs):
                    
                    steps_to_run = (epoch+1) * self.trainer.steps_per_loop - current_step

                    if steps_to_run <= 0:
                        continue
                    else:
                        if self.trainer.is_chief:
                            logging.info(f'@@@@@@@ start train at step {current_step} / epoch {epoch} to train {steps_to_run} steps@@@@@@@')
                        if not self.trainer.compiled:
                            train_op = self.trainer.make_train_op(eager_mode)
                            self.trainer.compiled = True
                            
                        self.registried_callbacks.before_epoch(steps_to_run)
                        for _ in range(steps_to_run):
                            train_op(next(iterator))
                            self.registried_callbacks.trigger_step()
                        
                        current_step = self.global_step.numpy()
                        train_metrics = {
                            m.name: m.result().numpy() for m in self.trainer.model.metrics
                        }
                        for m in self.trainer.model.metrics:
                            m.reset_state()
                        if current_step - last_evaluate_step >= self.trainer.steps_per_evaluate and self.trainer.is_chief:
                            #TODO support distributed evaluate later
                            # do evaluate
                            eval_results = self.run_evaluate(inference_op, self.trainer.val_dataset, eval_metrics)
                            last_evaluate_step = current_step
                        else:
                            eval_results = None

                        if self.trainer.is_chief:
                            logs = dict(train_metric = train_metrics)
                            if eval_results is not None:
                                logs['eval_metric'] = eval_results
                            logging.info(f'@@@@@@@@@@@@ logging train at step {current_step} / epoch {epoch} @@@@@@@@@@@@@')
                            # self._logger_epoch(train_metrics, eval_results)
                            logging.info(pprint.pformat(logs))
                            # logging.info(f'@@@@@@@@@@@@ logging callback at step {current_step} / epoch {epoch} @@@@@@@@@@@@@')
                            self.registried_callbacks.after_epoch(train_metrics, eval_results)
                            logging.info('@@@@@@@@@@@@ finished logging @@@@@@@@@@@@@')
            if self.trainer.is_chief:
                final_eval_results = self.run_evaluate(inference_op, self.trainer.val_dataset, eval_metrics)
            else:
                final_eval_results = None
            self.registried_callbacks.after_train(final_eval_results)

        except (StopTraining, tf.errors.OutOfRangeError) as e:
            logging.info("Training was stopped by exception {}.".format(str(e)))
        except KeyboardInterrupt:
            logging.info("Detected Ctrl-C and exiting main loop.")
            raise
        except Exception:
            logging.error("Training failed at global_step=", self.global_step.numpy())
            raise
        finally:
            # for i in self.trainer.model.trainable_variables:
            #     if not any([pattern in i.name for pattern in ["batch_normalization", "batchnorm", "bias", "beta"]]):
            #         print(i.name, tf.nn.l2_loss(i))
            self.registried_callbacks.after_run()

    def run_evaluate(self, op, dataset, metric):
        for data in dataset:
            try:
                inputs, labels = data
            except:
                inputs = data
                labels = None
            predictions = op(inputs)
            metric.update_state(predictions, labels)
        eval_results = metric.result()
        metric.reset_state()
        return eval_results

    def _register_callback(self, cb):
        """
        Register callbacks to the trainer.
        It can only be called before :meth:`Trainer.train()`.

        Args:
            cb (Callback or [Callback]): a callback or a list of callbacks

        Returns:
            succeed or not
        """
        
        assert not isinstance(self.registried_callbacks, CallbackList), \
            "Cannot register more callbacks after callbacklist was setup!"

        if isinstance(cb, (list, tuple)):
            for x in cb:
                self._register_callback(x)
        elif isinstance(cb, Callback):
            if not self.trainer.is_chief and cb.chief_only:
                logging.warn("Callback {} is chief-only, skipped.".format(str(cb)))
            else:
                self.registried_callbacks.append(cb)
        else:
            raise TypeError(f'{type(cb)} is not a valid callback type')
    
    def run(self, callbacks: List[Callback], eager_mode = False, xla = True):
        # trainer callbacks
        trainer_callback = self.trainer.trainer_callback()
        if trainer_callback is not None:
            if iterable(trainer_callback):
                for c in trainer_callback:
                    callbacks.append(c)
            else:
                callbacks.append(trainer_callback)
        self._register_callback(callbacks)
        self.registried_callbacks = CallbackList(self.registried_callbacks, self.trainer)
        self.main_loop(eager_mode, xla)

    def _logger_epoch(self, train_metric, eval_metric):
        logging.info(f'train_metrics:')
        for k, v in train_metric.items():
            logging.info(f' {k}: {v}')
        if eval_metric is not None:
            logging.info(f'eval_metrics:')
            for k, v in eval_metric.items():
                logging.info(f' {k}: {v}')
        
        
