import argparse
import os
import math
import sys

import tensorflow as tf
import tfcv
from tfcv import logger
from tfcv.distribute import MPI_is_distributed, MPI_local_rank, MPI_size
from tfcv.config import update_cfg, config as cfg
from tfcv.utils.lazy_import import LazyImport
hvd = LazyImport('horovod.tensorflow')

from tfcv.classification.tasks.base import ClassificationTask

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

def train(epochs):
    cfg.global_train_batch_size = cfg.train_batch_size * MPI_size()
    cfg.freeze()

    task = create_task(cfg)

    if cfg.data.use_dali:
        pass

def create_task(config):
    task = ClassificationTask(config)
    return task

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