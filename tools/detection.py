import os
import math
import shutil
import numpy as np
import tensorflow as tf
import yaml
import subprocess

import tfcv

from tfcv.config import update_cfg, setup_args, config as cfg
from tfcv.utils.default_args import RUN_PARSER
import signal

if __name__ == '__main__':
    arguments = RUN_PARSER.parse_args()

    config_file = arguments.config_file
    config_file = os.path.abspath(config_file)
    params = update_cfg(config_file)
    cfg.from_dict(params)
    setup_args(arguments, cfg)

    mode = arguments.mode
    
    workspace = arguments.workspace
    workspace = os.path.abspath(workspace)
    if arguments.reinitlize and 'train' in mode:
        shutil.rmtree(workspace, ignore_errors=True)
    if not os.path.isdir(workspace):
        os.makedirs(workspace)
    config_path = os.path.join(workspace, f'{arguments.mode}_config.yaml')
    with open(config_path, 'w') as fp:
        yaml.dump(cfg.to_dict(), fp, Dumper=yaml.CDumper)
    
   

    if mode == 'train':
        total_epochs = cfg.solver.epochs
        workspace = os.path.join(workspace, str(total_epochs))
        train_command = \
            f'python3 -m tfcv.detection.train --workspace {workspace} '\
            f'--config_file {config_path} --epochs {total_epochs}'

        ckpt = tf.train.latest_checkpoint(workspace)
        if ckpt != None:
            train_command += f' --initial_ckpt {ckpt}'


    elif mode == 'train_and_eval':
        total_epochs = cfg.solver.epochs
        evaluate_interval = cfg.solver.evaluate_interval
        if evaluate_interval == None:
            evaluate_interval = total_epochs

        epochs = np.arange(evaluate_interval, total_epochs+evaluate_interval, evaluate_interval)
        epochs[-1] = total_epochs

        last_epoch = 0
        for e in os.listdir(workspace):
            try:
                assert tf.train.latest_checkpoint(os.path.join(workspace, e)) != None
                e = int(e)
                if e > last_epoch:
                    last_epoch = e
            except:
                pass
        for target_epoch in epochs:
            if target_epoch < last_epoch:
                continue
            now_workspace = os.path.join(workspace, str(target_epoch))
            train_command = \
                f'python3 -m tfcv.detection.train --workspace {now_workspace} '\
                f'--config_file {config_path} --epochs {evaluate_interval}'
            if last_epoch > 0:
                restore_ckpt = tf.train.latest_checkpoint(os.path.join(workspace, str(last_epoch)))
                train_command += f' --initial_ckpt {restore_ckpt}'
            train_result = subprocess.run(train_command, shell=True)
            print('train script finished')
            if train_result.returncode == 0:
                print('train epoch finished')
            else:
                print('train failed: ', train_result.returncode)
                continue
            latest_ckpt = tf.train.latest_checkpoint(os.path.join(workspace, str(target_epoch)))
            result = os.path.join(workspace, str(target_epoch), 'eval_results.yaml')
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
            f' --checkpoints {" ".join(ckpts)} --results {os.path.join(workspace, "eval_results.yaml")}'
        eval_result = subprocess.run(eval_command, shell=True)
        