import argparse
import os
import math
import sys

import tensorflow as tf

import tfcv
from tfcv import logger
from tfcv.distribute import MPI_is_distributed, MPI_local_rank, MPI_size
from tfcv.exception import NanTrainLoss
from tfcv.hooks import LoggerHook, CheckpointAndBroadcastHook
from tfcv import HorovodTrainer, DefaultTrainer
from tfcv.config import update_cfg, config as cfg
from tfcv.datasets.coco.dataset import Dataset
from tfcv.detection.tasks.genelized_rcnn import GenelizedRCNNTask
from tfcv.schedules.learning_rate import PiecewiseConstantWithWarmupSchedule

from tfcv.utils.lazy_import import LazyImport
hvd = LazyImport('horovod.tensorflow')

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

def setup(config):
    if MPI_is_distributed():
        hvd.init()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus and MPI_is_distributed():
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
    tfcv.set_xla(config)
    tfcv.set_amp(config)

def train(epochs):
    cfg.global_train_batch_size = cfg.train_batch_size * MPI_size()
    cfg.freeze()
    
    task = create_task(cfg)

    dataset = Dataset()

    train_data = dataset.train_fn(task.train_preprocess, cfg.train_batch_size)

    total_steps = math.ceil(epochs * dataset.train_size / cfg.global_train_batch_size)

    global_step = tfcv.create_global_step()
    optimizer = create_optimizer(cfg, global_step, dataset.train_size)
    
    model = task.create_model()
    
    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer, global_step=global_step)
    metrics = create_metrics(cfg)
    hooks = [
        LoggerHook(logger),
        CheckpointAndBroadcastHook(
            checkpoint,
            cfg.workspace,
            cfg.checkpoint.name,
            cfg.solver.checkpoint_interval
        )
    ]
    trainer_cls = HorovodTrainer if MPI_is_distributed() else DefaultTrainer
    trainer = trainer_cls(
        cfg,
        global_step,
        model, 
        task, 
        optimizer, 
        metrics, 
        hooks
    )
    trainer.compile()
    return_code = trainer.train(total_steps, iter(train_data))
    sys.exit(return_code)
        
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

    backends = [logger.FileBackend(logger.Verbosity.INFO, os.path.join(workspace, f'train_log_rank{MPI_local_rank()}.txt'), True)]
    if not MPI_is_distributed() or MPI_local_rank() == 0:
        backends.append(logger.StdOutBackend(logger.Verbosity.INFO))

    logger.init(backends)

    train(arguments.epochs)