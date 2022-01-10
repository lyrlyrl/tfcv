import argparse
import os

from tfcv.config import update_cfg, config as cfg
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
    '--eval_id',
    type=int,
    required=True,
    nargs='+'
)

def eval(eval_ids):
    dataset = Dataset()
    eval_data = dataset.eval_fn(
        cfg.eval_batch_size)
    model = create_model()
    model.build(False)

    hooks = []
    evaluator = create_evaluator(model, hooks)

    evaluator.compile(False)

    for id in eval_ids:
        eval_once(evaluator, eval_data, id)

def eval_once(evaluator, eval_data, id):
    pass

def create_model():
    pass

def create_evaluator(model, hooks=[]):
    if cfg.meta_arch == 'genelized_rcnn':
        evaluator = GenelizedRCNNRunner(cfg, model, hooks)

if __name__ == '__main__':
    arguments = PARSER.parse_args()

    model_dir = arguments.model_dir
    config_path = os.path.join(model_dir, 'train_config.yaml')
    params = update_cfg(config_path)
    cfg.from_dict(params)
    cfg.model_dir = model_dir
    cfg.freeze()

    eval(arguments.eval_ids)