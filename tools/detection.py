import os
import math
import numpy as np
import tensorflow as tf
import yaml
import subprocess

import tfcv

from tfcv.config import update_cfg, setup_args, config as cfg
from tfcv.utils.default_args import RUN_PARSER


if __name__ == '__main__':
    arguments = RUN_PARSER.parse_args()

    config_file = arguments.config_file
    config_file = os.path.abspath(config_file)
    params = update_cfg(config_file)
    cfg.from_dict(params)
    setup_args(arguments, cfg)

    workspace = arguments.workspace
    workspace = os.path.abspath(workspace)
    if not os.path.isdir(workspace):
        os.makedirs(workspace)
    config_path = os.path.join(workspace, f'{arguments.mode}_config.yaml')
    with open(config_path, 'w') as fp:
        yaml.dump(cfg.to_dict(), fp, Dumper=yaml.CDumper)
    
    mode = arguments.mode
    if mode == 'train':
        total_epochs = cfg.solver.epochs
        evaluate_interval = cfg.solver.evaluate_interval
        if evaluate_interval == None:
            evaluate_interval = total_epochs

        epochs = np.arange(evaluate_interval, total_epochs+evaluate_interval, evaluate_interval)
        epochs[-1] = total_epochs

        last_epoch = -1
        for target_epoch in epochs:
            now_workspace = os.path.join(workspace, str(target_epoch))
            train_command = \
                f'python3 -m tfcv.detection.train --workspace {now_workspace} '\
                f'--config_file {config_path} --epochs evaluate_interval'
            if last_epoch > 0:
                restore_ckpt = tf.train.latest_checkpoint(os.path.join(workspace, str(last_epoch)))
                train_command += f' --initial_ckpt {restore_ckpt}'
            train_result = subprocess.run(train_command, shell=True)
            if train_result.returncode == 0:
                print('train epoch finished')
            else:
                print('train failed')
            latest_ckpt = tf.train.latest_checkpoint(os.path.join(workspace, str(target_epoch)))
            result = os.path.join(workspace, str(target_epoch), 'results.yaml')
            eval_command = \
                f'python3 -m tfcv.detection.evaluate --workspace {workspace} '\
                f'--config_file {config_path} --checkpoints {latest_ckpt} --results {result}'
            eval_result = subprocess.run(eval_command, shell=True)
            last_epoch = target_epoch

    elif mode == 'eval':
        ckpts = []
        for number in arguments.eval_numbers:
            assert os.path.isdir(os.path.join(workspace, str(number)))
            ckpt_path = tf.train.latest_checkpoint(os.path.join(workspace, str(number)))
            assert ckpt_path != None
            ckpts.append(ckpt_path)
        eval_command = \
            f'python3 -m tfcv.detection.evaluate --workspace {workspace} --config_file {config_path}'\
            f' --checkpoints {" ".join(ckpts)} --results {os.path.join(workspace, "results.yaml")}'
        