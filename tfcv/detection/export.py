import logging
import argparse
import os
import yaml

from tfcv.config import config as cfg
from tfcv.detection.runtime.run import export

PARSER = argparse.ArgumentParser(
    description='custom implementation of object detection models for TensorFlow 2.x',
    add_help=True)

PARSER.add_argument(
    '--model_dir',
    type=str,
    help='workspace dir',
    required=True
)

PARSER.add_argument(
    '--ckpt_number',
    type=int,
    default=None,
    help='workspace dir',
)

PARSER.add_argument(
    '--savedmodel_dir',
    type=str,
    default=None,
    help='export model to tf saved_model format'
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
    config_file = os.path.join(model_dir, f'export_config.yaml')
    with open(config_file, 'r') as fp:
        params = yaml.load(fp, Loader=yaml.CLoader)
    cfg.from_dict(params)

    cfg.model_dir = model_dir
    savedmodel_dir = arguments.savedmodel_dir
    if not savedmodel_dir:
        savedmodel_dir = os.path.join(model_dir, 'savedmodel')
    export(savedmodel_dir, arguments.ckpt_number)