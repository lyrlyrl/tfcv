import subprocess
import logging
import os
import shutil
import math

import yaml
from tfcv.config import update_cfg, setup_args, config as cfg
from tfcv.utils.default_args import RUN_PARSER

if __name__ == '__main__':
    arguments = RUN_PARSER.parse_args()

    config_file = arguments.config_file
    config_file = os.path.abspath(config_file)
    params = update_cfg(config_file)
    cfg.from_dict(params)
    setup_args(arguments, cfg)

    model_dir = arguments.model_dir
    model_dir = os.path.abspath(model_dir)
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    config_path = os.path.join(model_dir, f'{arguments.mode}_config.yaml')
    with open(config_path, 'w') as fp:
        yaml.dump(cfg.to_dict(), fp, Dumper=yaml.CDumper)

    if arguments.model == 'train':
        total_epochs = cfg.solver.epochs
        evaluate_interval = cfg.solver.evaluate_interval

        run_ids = list(range(math.ceil(total_epochs/evaluate_interval)))

        checkpoint_dir = os.path.join(model_dir, 'checkpoint')
        if arguments.reinitlize:
            shutil.rmtree(checkpoint_dir, ignore_errors=True)
        historys = os.listdir(checkpoint_dir)
        run_id = -1
        for i in historys:
            try:
                i = int(i)
                assert i in run_ids
            except:
                shutil.rmtree(os.path.join(checkpoint_dir, str(i)))
                logging.info(f'unuseful file or dir {os.path.join(checkpoint_dir, str(i))} remove')
                continue
            if i > run_id:
                run_id = i
        run_id += 1
        while True:
            if run_id not in run_ids:
                break
            train_command = f'python3 -m tfcv.runtime.train_detection --model_dir {model_dir} --run_id {str(run_id)}'
            if len(arguments.gpu_ids) > 1:
                n = len(arguments.gpu_ids)
                prefix = f'CUDA_VISIBLE_DEVICES={",".join(arguments.gpu_ids)} horovodrun -np {n} -H localhost:{n} '
                train_command = prefix + train_command
            train_result = subprocess.run(train_command, capture_output=True)
            if train_result.returncode == 0:
                logging.info('train epoch finished')
            else:
                pass
            eval_command = f'python3 -m tfcv.runtime.evaluate_detection --model_dir {model_dir} --eval_id {str(run_id)}'
            eval_result = subprocess.run(eval_command, capture_output=True)
            if eval_result.returncode == 0:
                logging.info('eval of epoch finished')
            else:
                pass
            run_id += 1
        logging.info('train finished')