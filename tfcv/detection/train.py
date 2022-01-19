import argparse
import os
import logging
import math

import tensorflow as tf

import tfcv
from tfcv import logger
from tfcv.exception import NanTrainLoss
from tfcv import DefaultTrainer
from tfcv.config import update_cfg, config as cfg
from tfcv.datasets.coco.dataset import Dataset
from tfcv.detection.tasks.genelized_rcnn import GenelizedRCNNTask
from tfcv.schedules.learning_rate import PiecewiseConstantWithWarmupSchedule

PARSER = argparse.ArgumentParser(
    description='as child process'
)
PARSER.add_argument(
    '--workspace',
    type=str,
    required=True
)
PARSER.add_argument(
    '--config_file',
    type=str,
    required=True
)
PARSER.add_argument(
    '--epochs',
    type=int,
    required=True
)
PARSER.add_argument(
    '--initial_ckpt',
    type=str,
)

def setup(config):
    tfcv.set_xla(config)

    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    tfcv.set_amp(config)

def train(epochs, initial_ckpt=None):
    strategy = tfcv.get_strategy(cfg)
    # cfg.replicas = strategy.num_replicas_in_sync
    cfg.global_train_batch_size = cfg.train_batch_size * strategy.num_replicas_in_sync
    cfg.freeze()
    
    dataset = Dataset()
    train_data = dataset.train_fn(cfg.global_train_batch_size, strategy=strategy)
    total_steps = math.ceil(epochs * dataset.train_size / cfg.global_train_batch_size)

    with strategy.scope():
        global_step = tfcv.create_global_step()
        optimizer = create_optimizer(cfg, global_step, dataset.train_size)
        task = create_task(cfg)
        model = task.create_model()
        
        checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer, global_step=global_step)
        if initial_ckpt != None:
            checkpoint.restore(initial_ckpt)
        metrics = create_metrics(cfg)
        trainer = DefaultTrainer(
            cfg, global_step, model, task, optimizer, metrics
        )

        trainer.compile()
        try:
            trainer.train(total_steps, iter(train_data))
            ckpt_path = os.path.join(cfg.workspace, cfg.checkpoint.name)
            checkpoint.save(ckpt_path)
        except NanTrainLoss:
            success = 'train loss nan'
        
        logger.finalize(global_step.numpy(), success)
        
def create_task(config):
    if config.meta_arch == 'genelized_rcnn':
        task = GenelizedRCNNTask(config)
    else:
        raise
    return task

def create_metrics(config):

    if config.meta_arch == 'genelized_rcnn':
        metrics = [
                tf.keras.metrics.Mean(name='rpn_score_loss'),
                tf.keras.metrics.Mean(name='rpn_box_loss'),
                tf.keras.metrics.Mean(name='fast_rcnn_class_loss'),
                tf.keras.metrics.Mean(name='fast_rcnn_box_loss'),
                tf.keras.metrics.Mean(name='l2_regularization_loss')
            ]
            
        if config.include_mask:
            metrics.append(tf.keras.metrics.Mean(name='mask_rcnn_loss'))

    return metrics

def create_optimizer(config, global_step, train_size):
    if config.meta_arch == 'genelized_rcnn':
        learning_rate = PiecewiseConstantWithWarmupSchedule(
            global_step,
            init_value=config.optimization.init_learning_rate,
            # scale boundaries from epochs to steps
            boundaries=[
                int(b * train_size / config.global_train_batch_size)
                for b in config.optimization.learning_rate_boundaries
            ],
            values=config.optimization.learning_rate_values,
            # scale only by local BS as distributed strategy later scales it by number of replicas
            scale=config.train_batch_size
        )
        optimizer = tf.keras.optimizers.SGD(
            learning_rate=learning_rate,
            momentum=config.optimization.momentum
        )
    if config.amp:
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer, dynamic=True)
    return optimizer

if __name__ == '__main__':
    arguments = PARSER.parse_args()

    workspace = arguments.workspace
    if not os.path.isdir(workspace):
        os.makedirs(workspace)
    params = update_cfg(arguments.config_file)
    cfg.from_dict(params)
    cfg.workspace = workspace

    setup(cfg)

    logger.init(
        [
            logger.StdOutBackend(logger.Verbosity.INFO),
            logger.FileBackend(logger.Verbosity.DEBUG, os.path.join(workspace, 'train_log.txt'), False)
        ]
    )
    train(arguments.epochs, arguments.initial_ckpt)