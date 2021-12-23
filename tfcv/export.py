import argparse
import os
import subprocess
import yaml

from tfcv.train import update_cfg
from tfcv.config import config as cfg
PARSER = argparse.ArgumentParser(
    description='export pretreaned models',
    add_help=True)

PARSER.add_argument(
    '--model_dir',
    type=str,
    default=None,
    help='workspace dir',
    required=True
)

PARSER.add_argument(
    '--savedmodel_dir',
    type=str,
    default=None,
    help='export model to tf saved_model format'
)

PARSER.add_argument(
    '--ckpt_number',
    type=int,
    help='checkpoint number to export',
    required=True
)

PARSER.add_argument(
    '--config_file',
    type=str,
    default=None,
    help='config file',
    required=True
)

PARSER.add_argument(
    '--precision',
    type=str,
    default='fp32',
    choices=['fp32', 'fp16', 'int8'],
    help='workspace dir',
)

if __name__ == '__main__':
    arguments = PARSER.parse_args()
    print(arguments)
    config_file = arguments.config_file
    config_file = os.path.abspath(config_file)
    params = update_cfg(config_file)
    cfg.from_dict(params)
    cfg.freeze()

    model_dir = arguments.model_dir
    model_dir = os.path.abspath(model_dir)
    assert os.path.isdir(model_dir)
    config_path = os.path.join(model_dir, f'export_config.yaml')
    with open(config_path, 'w') as fp:
        yaml.dump(cfg.to_dict(), fp, Dumper=yaml.CDumper)
    
    if cfg.task == 'detection':
        main_path = 'tfcv.detection.export'
        cmd = (f'python -m {main_path}'
            f' --model_dir {model_dir}')

        if arguments.savedmodel_dir:
            cmd += f' --savedmodel_dir {arguments.savedmodel_dir}'

        if arguments.ckpt_number:
            cmd += f' --ckpt_number {arguments.ckpt_number}'
        
        subprocess.call(cmd, shell=True)