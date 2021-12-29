import logging
import os
import math
import numpy as np
import tensorflow as tf
from tensorflow.python.framework.errors_impl import NotFoundError
import yaml

import tfcv
from tfcv.hooks.base import Hook, HookList
from tfcv.hooks.test_hook import TestHook
from tfcv.datasets.coco.dataset import Dataset
from tfcv.models.genelized_rcnn import GenelizedRCNN
from tfcv.schedules.learning_rate import PiecewiseConstantWithWarmupSchedule
from tfcv.runners.faster_rcnn import FasterRCNNTrainer, FasterRCNNExporter

from tfcv.config import update_cfg, setup_args, config as cfg
from tfcv.utils.default_args import TRAIN_PARSER

def setup():
    tfcv.set_xla(cfg)

    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    tfcv.set_amp(cfg)

def train_and_evaluate(export_to_savedmodel=False):
    setup()
    dataset = Dataset()
    strategy = tfcv.get_strategy(cfg)

    cfg.replicas = strategy.num_replicas_in_sync
    cfg.global_train_batch_size = cfg.train_batch_size * cfg.replicas
    cfg.global_eval_batch_size = cfg.eval_batch_size * cfg.replicas
    cfg.freeze()
    logging.info(f'Distributed Strategy is activated for {cfg.replicas} device(s)')

    train_data = dataset.train_fn(batch_size=cfg.global_train_batch_size)
    eval_data = dataset.eval_fn(batch_size=cfg.global_eval_batch_size)

    with strategy.scope():

        learning_rate = PiecewiseConstantWithWarmupSchedule(
            init_value=cfg.optimization.init_learning_rate,
            # scale boundaries from epochs to steps
            boundaries=[
                int(b * dataset.train_size / cfg.global_train_batch_size)
                for b in cfg.optimization.learning_rate_boundaries
            ],
            values=cfg.optimization.learning_rate_values,
            # scale only by local BS as distributed strategy later scales it by number of replicas
            scale=cfg.train_batch_size
        )

        optimizer = tf.keras.optimizers.SGD(
            learning_rate=learning_rate,
            momentum=cfg.optimization.momentum
        )

        model = create_model()
        checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
        ckpt_path = os.path.join(cfg.model_dir, cfg.checkpoint.subdir, cfg.checkpoint.name)
        initialize(checkpoint)

        metrics = create_metrics()
        hooks = HookList(hooks=[
            TestHook('test')
        ])
        trainer = create_trainer(model, optimizer, metrics, hooks)

        dist_train_dataset = strategy.experimental_distribute_dataset(train_data)
        dist_eval_dataset = strategy.experimental_distribute_dataset(eval_data)
        trainer.compile()
        train_iter = iter(dist_train_dataset)

        total_steps = math.ceil(cfg.solver.epochs * dataset.train_size / cfg.global_train_batch_size)
        steps_per_epoch = int(total_steps * cfg.solver.evaluate_interval / cfg.solver.epochs)
        current_step = optimizer.iterations.numpy()

        eval_results = {}
        best_results = {}
        eval_results_dir = os.path.join(cfg.model_dir, 'eval_results.yaml')

        for i in np.arange(0, cfg.solver.epochs, cfg.solver.evaluate_interval):
            i = np.round(i, 3)
            if i * steps_per_epoch < current_step:
                continue
            step_to_train = min(steps_per_epoch, total_steps - i * steps_per_epoch)
            try:
                trainer.train(step_to_train, train_iter, epoch_number=i+cfg.solver.evaluate_interval)
            except tfcv.NanTrainLoss as e:
                logging.warn(e)
            checkpoint.save(ckpt_path)
            eval_result = trainer.evaluate(dist_eval_dataset)
            eval_result['save_count'] = checkpoint.save_count.numpy().tolist()
            eval_results[str(i)] = eval_result
            with open(eval_results_dir, 'w') as fp:
                yaml.dump(eval_results, fp, Dumper=yaml.CDumper)
        
        if export_to_savedmodel:
            pass

def evaluate(eval_number):
    setup()
    dataset = Dataset()
    if cfg.num_gpus > 1:
        strategy = tf.distribute.MirroredStrategy(
            devices=["device:GPU:%d" % i for i in range(cfg.num_gpus)]
        )
    elif cfg.num_gpus == 1:
        strategy = tf.distribute.OneDeviceStrategy('device:GPU:0')
    else:
        strategy = tf.distribute.OneDeviceStrategy('device:CPU:0')
    cfg.replicas = strategy.num_replicas_in_sync
    cfg.global_eval_batch_size = cfg.eval_batch_size * cfg.replicas
    cfg.freeze()
    logging.info(f'Distributed Strategy is activated for {cfg.replicas} device(s)')
    eval_data = dataset.eval_fn(batch_size=cfg.global_eval_batch_size)

    with strategy.scope():

        model = create_model()
        checkpoint = tf.train.Checkpoint(model=model)

        trainer = create_trainer(model)

        dist_eval_dataset = strategy.experimental_distribute_dataset(eval_data)
        trainer.compile(train=False)

        eval_results = {}
        eval_results_dir = os.path.join(cfg.model_dir, 'run_eval_results.yaml')

        for i in eval_number:
            ckpt_path = os.path.join(cfg.model_dir, cfg.checkpoint.subdir, f'{cfg.checkpoint.name}-{i}')
            try:
                checkpoint.restore(ckpt_path).expect_partial()
                eval_result = trainer.evaluate(dist_eval_dataset)
                eval_results[i] = eval_result
            except NotFoundError:
                logging.error(f'cant find checkpoint {ckpt_path}')
            finally:
                with open(eval_results_dir, 'w') as fp:
                    yaml.dump(eval_results, fp, Dumper=yaml.CDumper)

def export(savedmodel_dir, ckpt_number=None):
    model = create_model()
    ckpt = tf.train.Checkpoint(model=model)
    if ckpt_number != None:
        logging.info('export version specific model')
        ckpt_path = os.path.join(cfg.model_dir, cfg.checkpoint.subdir, f'{cfg.checkpoint.name}-{ckpt_number}')
    else:
        logging.info('export best model')
    ckpt.restore(ckpt_path).expect_partial()

    exporter = create_exporter(model)
    tf.saved_model.save(
        exporter, 
        savedmodel_dir, 
        signatures=exporter.inference_from_tensor.get_concrete_function(tf.TensorSpec([cfg.inference_batch_size]+list(cfg.data.export_image_size)+[3], tf.uint8))
        )

def create_model():
    if cfg.meta_arch == 'genelized_rcnn':
        model = GenelizedRCNN(cfg)

    return model

def initialize(checkpoint):
    ckpt_path = os.path.join(cfg.model_dir, cfg.checkpoint.subdir)
    checkpoint_path = tf.train.latest_checkpoint(ckpt_path)
    if checkpoint_path is None:
        logging.info(f"No checkpoint was found in: {ckpt_path}")
        return
    checkpoint.restore(checkpoint_path).assert_consumed()
    logging.info(f"Loaded weights from checkpoint: {checkpoint_path}")

def create_metrics():

    if cfg.meta_arch == 'genelized_rcnn':
        metrics = [
                tf.keras.metrics.Mean(name='rpn_score_loss'),
                tf.keras.metrics.Mean(name='rpn_box_loss'),
                tf.keras.metrics.Mean(name='fast_rcnn_class_loss'),
                tf.keras.metrics.Mean(name='fast_rcnn_box_loss'),
                tf.keras.metrics.Mean(name='l2_regularization_loss')
            ]
            
        if cfg.include_mask:
            metrics.append(tf.keras.metrics.Mean(name='mask_rcnn_loss'))

    return metrics

def create_trainer(model, optimizer=None, metrics=[], hooks=[]):
    if cfg.meta_arch == 'genelized_rcnn':
        return FasterRCNNTrainer(cfg, model, optimizer, metrics, hooks)

def create_exporter(model):
    if cfg.meta_arch == 'genelized_rcnn':
        return FasterRCNNExporter(model, cfg)

if __name__ == '__main__':
    arguments = TRAIN_PARSER.parse_args()

    # setup logging
    logging.basicConfig(
        # level=logging.DEBUG if params.verbose else logging.INFO,
        level=logging.INFO,
        format='{asctime} {levelname:.1} {name:15} {message}',
        style='{'
    )

    # remove custom tf handler that logs to stderr
    logging.getLogger('tensorflow').setLevel(logging.WARN)
    logging.getLogger('tensorflow').handlers.clear()

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
    cfg.model_dir = model_dir
    if arguments.mode == 'train':
        train_and_evaluate()
    else:
        assert arguments.eval_number
        eval_number = ' '.join(str(s) for s in arguments.eval_number)
        evaluate(eval_number)
