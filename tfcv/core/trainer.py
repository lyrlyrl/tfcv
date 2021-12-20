import math

import tensorflow as tf
import logging

import time

__all__ = ['Trainer', 'merge_replica_results']

def merge_replica_results(strategy, inputs):
    dist_values = strategy.experimental_local_results(inputs)
    def _merge(*args):
        return tf.concat(args, 0)
    return tf.nest.map_structure(_merge, *dist_values)

class Trainer(tf.Module):

    def __init__(
            self, 
            params,
            model: tf.keras.Model,
            optimizer,
            metrics=[]) -> None:
        self._params = params

        self._model = model
        self._optimizer = optimizer
        self._metrics = metrics

        self._train_loss = tf.keras.metrics.Mean(name='train_loss')

        self._train_loop_fn = None
        self._inference_op = None
        
        self._train_timer = 0

    def compile(self):
        strategy = tf.distribute.get_strategy()
        train_step_fn = tf.function(self.train_step)
        def dist_train_step(iterator):
            strategy.run(
                train_step_fn,
                args=(next(iterator),)
            )
        dist_train_step = tf.function(dist_train_step)
        def train_loop(iterator, steps):
            try:
                with tf.experimental.async_scope():
                    for _ in range(steps):
                        dist_train_step(iterator)
            except tf.errors.OutOfRangeError:
                tf.experimental.async_clear_error()
        self._train_loop_fn = train_loop
        if not self._inference_op:
            def dist_inference_op(dist_inputs):
                per_replica_predictions = strategy.run(self.inference_step, args=(dist_inputs, ))
                predictions = merge_replica_results(strategy, per_replica_predictions)
                return predictions
            self._inference_op = tf.function(dist_inference_op)


    def train(self, num_steps, train_iterator, current_step = 0):
        if not self._train_loop_fn:
            raise 
        assert num_steps > 0
        num_loops = math.ceil(num_steps / self._params.solver.steps_per_loop)

        for loop_number in range(num_loops):
            steps_to_run = (loop_number+1) * self._params.solver.steps_per_loop - current_step            
            logging.info(f'train loop start at {current_step}')
            self.train_loop_begin()
            self._train_loop_fn(train_iterator, steps_to_run)
            self.train_loop_end()
            current_step += steps_to_run

    def train_step(self, inputs):
        with tf.GradientTape() as tape:
            model_outputs = self._model(inputs[0], training=True)
            raw_loss = tf.math.reduce_sum(self._model.losses)
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

    def inference_step(self, inputs):
        model_outputs = self._model(inputs, training=False)
        return model_outputs
    
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

        logging.info(f'train_throuput: {throuput}')

        logging.info(f'train_loss: {self._train_loss.result().numpy()}')

        for metric in self._metrics:
            logging.info(f'{metric.name}: {metric.result().numpy()}')
        
        for metric in self._model.metrics:
            logging.info(f'{metric.name}: {metric.result().numpy()}')
    
    def eval_begin(self):
        pass

    def eval_end(self):
        pass