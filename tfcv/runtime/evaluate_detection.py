import argparse
import os
import logging
import math

import tensorflow as tf

import tfcv
from tfcv import logger
from tfcv.config import update_cfg, config as cfg
from tfcv.datasets.coco.dataset import Dataset
from tfcv.models.genelized_rcnn import GenelizedRCNN
from tfcv.schedules.learning_rate import PiecewiseConstantWithWarmupSchedule
from tfcv.runners.faster_rcnn import FasterRCNNTrainer

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
    default=None,
    nargs='+',
    required=True
)

def evaluate(eval_number):
    setup()
    dataset = Dataset()
    if cfg.num_gpus > 1:
        strategy = tf.distribute.MirroredStrategy(
            devices=["device:GPU:%d" % i for i in range(cfg.num_gpus)]
        )
    elif cfg.num_gpus == 1:
        strategy = tf.distribute.OneDeviceStrategy('device:GPU:0')
    else:
        strategy = tf.distribute.OneDeviceStrategy('device:CPU:0')
    cfg.replicas = strategy.num_replicas_in_sync
    cfg.global_eval_batch_size = cfg.eval_batch_size * cfg.replicas
    cfg.freeze()
    logging.info(f'Distributed Strategy is activated for {cfg.replicas} device(s)')
    eval_data = dataset.eval_fn(batch_size=cfg.global_eval_batch_size)

    with strategy.scope():

        model = create_model()
        checkpoint = tf.train.Checkpoint(model=model)

        trainer = create_trainer(model)

        dist_eval_dataset = strategy.experimental_distribute_dataset(eval_data)
        trainer.compile(train=False)

        eval_results = {}
        eval_results_dir = os.path.join(cfg.model_dir, 'run_eval_results.yaml')

        for i in eval_number:
            ckpt_path = os.path.join(cfg.model_dir, cfg.checkpoint.subdir, f'{cfg.checkpoint.name}-{i}')
            try:
                checkpoint.restore(ckpt_path).expect_partial()
                eval_result = trainer.evaluate(dist_eval_dataset)
                eval_results[i] = eval_result
            except NotFoundError:
                logging.error(f'cant find checkpoint {ckpt_path}')
            finally:
                with open(eval_results_dir, 'w') as fp:
                    yaml.dump(eval_results, fp, Dumper=yaml.CDumper)