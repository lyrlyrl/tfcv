import logging
import argparse
import os
import yaml

from tfcv.config import config as cfg
from tfcv.detection.runtime.run import ctl_train, evaluate

PARSER = argparse.ArgumentParser(
    description='custom implementation of object detection models for TensorFlow 2.x',
    add_help=True)

PARSER.add_argument(
    'mode',
    type=str,
    metavar='MODE',
    choices=['train', 'eval'],
    help='run mode')

PARSER.add_argument(
    '--model_dir',
    type=str,
    default=None,
    help='workspace dir',
    required=True
)

if __name__ == '__main__':
    # setup params
    arguments = PARSER.parse_args()
    # setup logging
    logging.basicConfig(
        # level=logging.DEBUG if params.verbose else logging.INFO,
        level=logging.INFO,
        format='{asctime} {levelname:.1} {name:15} {message}',
        style='{'
    )

    # remove custom tf handler that logs to stderr
    logging.getLogger('tensorflow').setLevel(logging.INFO)
    logging.getLogger('tensorflow').handlers.clear()

    model_dir = arguments.model_dir
    config_file = os.path.join(model_dir, f'{arguments.mode}_config.yaml')
    with open(config_file, 'r') as fp:
        params = yaml.load(fp, Loader=yaml.CLoader)
    cfg.from_dict(params)
    cfg.model_dir = model_dir
    if arguments.mode == 'train':
        ctl_train()
    else:
        evaluate()