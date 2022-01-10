import os
import argparse
import math
import tensorflow as tf
import shutil
import glob
import logging

from tfcv import logger
from tfcv.distribute import *
from tfcv.config import update_cfg, config as cfg

from tfcv.utils.lazy_import import LazyImport
hvd = LazyImport('horovod.tensorflow')

from tfcv.schedules.learning_rate import PiecewiseConstantWithWarmupSchedule
from tfcv.datasets.coco.dataset import Dataset
from tfcv.runners.genelized_rcnn import GenelizedRCNNRunner
PARSER = argparse.ArgumentParser(
    description='as child process'
)
PARSER.add_argument(
    '--model_dir',
    type=str,
    required=True
)
PARSER.add_argument(
    '--run_id',
    type=int,
    required=True
)

def train(run_id):
    dataset = Dataset()
    train_data = dataset.train_fn(
        cfg.train_batch_size, 
        shard=(MPI_size(), MPI_local_rank()) if MPI_is_distributed() else None)

    learning_rate = PiecewiseConstantWithWarmupSchedule(
        init_value=cfg.optimization.init_learning_rate,
        # scale boundaries from epochs to steps
        boundaries=[
            int(b * dataset.train_size / cfg.global_train_batch_size)
            for b in cfg.optimization.learning_rate_boundaries
        ],
        values=cfg.optimization.learning_rate_values,
        scale=cfg.global_train_batch_size
    )

    optimizer = tf.keras.optimizers.SGD(
        learning_rate=learning_rate,
        momentum=cfg.optimization.momentum
    )

    model = create_model()
    model.build(True)
    
    checkpoint = initialize(model, optimizer, run_id)

    hooks = []
    metrics = create_metrics()
    trainer = create_trainer(model, optimizer, metrics, hooks)

    trainer.compile(True)
    train_iter = iter(train_data)
    total_steps = math.ceil(cfg.solver.epochs * dataset.train_size / cfg.global_train_batch_size)

    trainer.train(total_steps, train_iter)

    step = optimizer.iterations.numpy()

    model.save_weights(
        os.path.join(cfg.model_dir, 'checkpoint', str(run_id), f'{model.name}-{str(step)}.npz')
    )
    checkpoint.write(os.path.join(cfg.model_dir, 'checkpoint', 'optimizer', 'opt'))


def create_model():
    pass

def create_metrics():
    metrics = {'l2_loss': tf.keras.metrics.Mean(name = 'l2_loss')}
    if cfg.meta_arch == 'genelized_rcnn':
        metrics.update({
            'fast_rcnn_class_loss': tf.keras.metrics.Mean(name = 'fast_rcnn_class_loss'),
            'fast_rcnn_box_loss': tf.keras.metrics.Mean(name = 'fast_rcnn_box_loss'),
            'rpn_score_loss': tf.keras.metrics.Mean(name = 'rpn_score_loss'),
            'rpn_box_loss': tf.keras.metrics.Mean(name = 'rpn_box_loss')
        })
        if cfg.include_mask:
            metrics['mask_loss'] = tf.keras.metrics.Mean(name = 'mask_loss')
    return metrics
def create_trainer(model, optimizer, metrics={}, hooks=[]):
    if cfg.meta_arch == 'genelized_rcnn':
        trainer = GenelizedRCNNRunner(cfg, model, optimizer, metrics, hooks)

def initialize(model, opt, run_id):
    all_weight_dir = os.path.join(cfg.model_dir, 'checkpoint')

    checkpoint = tf.train.Checkpoint(optimizer = opt)

    if run_id > 0:
        weight_dir = os.path.join(all_weight_dir, str(run_id-1))
        np_path_pattern = os.path.join(weight_dir, f'{model.name}-*.npz')
        path = glob.glob(np_path_pattern)[0]
        model.load_weights(path)

        if os.path.isdir(os.path.join(weight_dir, 'optimizer')):
            opt_path = tf.train.latest_checkpoint(
                os.path.join(weight_dir, 'optimizer'), latest_filename=None
            )
            checkpoint.read(opt_path)

    return checkpoint

if __name__ == '__main__':
    arguments = PARSER.parse_args()

    if MPI_is_distributed():
        hvd.init()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus and MPI_is_distributed():
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
    
    model_dir = arguments.model_dir
    config_path = os.path.join(model_dir, 'train_config.yaml')
    params = update_cfg(config_path)
    cfg.from_dict(params)
    cfg.model_dir = model_dir
    if MPI_is_distributed():
        cfg.global_train_batch_size = cfg.train_batch_size * hvd.size()
    else:
        cfg.global_train_batch_size = cfg.train_batch_size
    cfg.freeze()
    
    log_path = os.path.join(model_dir, 'logs', str(arguments.run_id))
    os.makedirs(log_path, exist_ok=True)
    if MPI_is_distributed():
        log_file = os.path.join(log_path, 'train_log.txt')
    else:
        log_file = os.path.join(log_path, 'train_log_rank{}.txt'.format(MPI_local_rank()))

    logger_backends = [logger.FileBackend(logger.Verbosity.DEBUG, file_path=log_file, proceed=False)]
    if not MPI_is_distributed() or MPI_local_rank()==0:
        logger_backends.append(logger.StdOutBackend(logger.Verbosity.INFO))
    logger.init(logger_backends)

    train(arguments.run_id)

