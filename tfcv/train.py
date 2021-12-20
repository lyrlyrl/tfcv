import yaml
import subprocess
import collections
import sys
import os
from tfcv.config import config as cfg
from tfcv.utils.default_args import PARSER

def dict_update(orig_dict, new_dict):
    for key, val in new_dict.items():
        if isinstance(val, collections.abc.Mapping):
            tmp = dict_update(orig_dict.get(key, {}), val)
            orig_dict[key] = tmp
        else:
            orig_dict[key] = new_dict[key]
    return orig_dict

def update_cfg(config_file):
    assert os.path.isfile(config_file)
    with open(config_file, 'r') as f:
        params = yaml.load(f, Loader=yaml.CLoader)
    if '_base' in params:
        new_config_file = os.path.normpath(os.path.join(config_file, '..', params['_base']))
        params = dict_update(update_cfg(new_config_file), params)
        # params.update(update_cfg(new_config_file))
        del params['_base']
    return params

def setup_args(arguments):
    cfg.train_batch_size = arguments.train_batch_size
    cfg.eval_batch_size = arguments.eval_batch_size
    cfg.seed = arguments.seed
    cfg.xla = arguments.xla
    cfg.amp = arguments.amp
    cfg.epochs = arguments.epochs
    cfg.steps_per_loop = arguments.steps_per_loop
    cfg.eval_samples = arguments.eval_samples

if __name__ == '__main__':
    arguments = PARSER.parse_args()

    config_file = arguments.config_file
    config_file = os.path.abspath(config_file)
    params = update_cfg(config_file)
    cfg.from_dict(params)
    if not arguments.strict_config:
        setup_args(arguments)
    cfg.freeze()

    model_dir = arguments.model_dir
    model_dir = os.path.abspath(model_dir)
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    config_path = os.path.join(model_dir, f'{arguments.mode}_config.yaml')
    with open(config_path, 'w') as fp:
        yaml.dump(cfg.to_dict(), fp, Dumper=yaml.CDumper)

    if arguments.task == 'detection':
        main_path = 'tfcv.detection.main'
        if arguments.mode == 'train':
            subprocess.call(
                (f'python -m {main_path}'
                f' train'
                f' --model_dir {model_dir}'),
                shell=True
            )
        else:
            subprocess.run(
                f'python -m {main_path}'
                f' eval',
                f' --model_dir {model_dir}'
            )
    
