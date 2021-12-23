import yaml
import subprocess
import collections
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
    if arguments.seed:
        cfg.seed = arguments.seed
    cfg.xla = arguments.xla
    cfg.amp = arguments.amp
    cfg.steps_per_loop = arguments.steps_per_loop
    cfg.num_gpus = arguments.num_gpus
    if arguments.config_override:
        cfg.update_args(arguments.config_override)

if __name__ == '__main__':
    arguments = PARSER.parse_args()
    print(arguments)
    config_file = arguments.config_file
    config_file = os.path.abspath(config_file)
    params = update_cfg(config_file)
    cfg.from_dict(params)
    setup_args(arguments)
    cfg.freeze()

    model_dir = arguments.model_dir
    model_dir = os.path.abspath(model_dir)
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    config_path = os.path.join(model_dir, f'{arguments.mode}_config.yaml')
    with open(config_path, 'w') as fp:
        yaml.dump(cfg.to_dict(), fp, Dumper=yaml.CDumper)
    num_gpus = arguments.num_gpus

    if cfg.task == 'detection':
        main_path = 'tfcv.detection.train'
        if arguments.mode == 'train':
            cmd_train = (f'python -m {main_path}'
                f' train'
                f' --model_dir {model_dir}')
            if arguments.export_to_savedmodel:
                cmd_train += ' --export_to_savedmodel'
            code = subprocess.call(
                cmd_train,
                shell=True
            )
        else:
            assert arguments.eval_number
            eval_number = ' '.join(str(s) for s in arguments.eval_number)
            code = subprocess.call(
                (f'python -m {main_path}'
                f' eval'
                f' --model_dir {model_dir}'
                f' --eval_number {eval_number}'),
                shell=True
            )
    exit(code)
