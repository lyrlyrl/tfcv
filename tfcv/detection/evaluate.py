import argparse
import os
import logging
import math
import yaml

import tensorflow as tf

import tfcv
from tfcv import logger
from tfcv import Predictor
from tfcv.config import update_cfg, config as cfg
from tfcv.datasets.coco.dataset import Dataset
from tfcv.evaluate.metric import COCOEvaluationMetric, process_predictions
from tfcv.detection.train import create_task, setup

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
    '--results',
    type=str,
    required=True
)
PARSER.add_argument(
    '--checkpoints',
    type=str,
    default=None,
    nargs='+',
    required=True
)

def evaluate(ckpts, results):
    dataset = Dataset()
    cfg.global_eval_batch_size = cfg.eval_batch_size
    cfg.freeze()
    eval_data = dataset.eval_fn(batch_size=cfg.global_eval_batch_size)

    eval_results = {}

    coco_metric = COCOEvaluationMetric(
        os.path.expanduser(os.path.join(cfg.data.dir, cfg.data.val_json)), cfg.include_mask)

    task = create_task(cfg)
    model = task.create_model()

    checkpoint = tf.train.Checkpoint(model=model)

    predictor = Predictor(cfg, model, task)
    predictor.compile()
    
    for ckpt in ckpts:
        checkpoint.restore(ckpt).expect_partial()
        outputs = []
        for inputs in eval_data:
            outputs.append(predictor.predict_batch(inputs))
            
        def _merge(*args):
            return tf.concat(args, 0).numpy()
        outputs = tf.nest.map_structure(_merge, *outputs)
        predictions = process_predictions(outputs)
        metric = coco_metric.predict_metric_fn(predictions)
        eval_results[ckpt] = tf.nest.map_structure(tfcv.autocast, metric)
        
    with open(results, 'w') as fp:
        yaml.dump(eval_results, fp, Dumper=yaml.CDumper)

if __name__ == '__main__':
    arguments = PARSER.parse_args()

    workspace = arguments.workspace
    cfg.workspace = workspace
    if not os.path.isdir(workspace):
        os.makedirs(workspace)
    params = update_cfg(arguments.config_file)
    cfg.from_dict(params)

    setup(cfg)

    logger.init(
        [
            logger.StdOutBackend(logger.Verbosity.INFO),
            logger.FileBackend(logger.Verbosity.INFO, os.path.join(workspace, 'evaluate_log.txt'), False)
        ]
    )
    evaluate(arguments.checkpoints, arguments.results)