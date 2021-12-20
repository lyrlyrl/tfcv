import logging
import os
import numpy as np
import tensorflow as tf
import yaml

from tfcv.config import config as cfg

from tfcv.detection.dataset.dataset import Dataset
from tfcv.detection.modeling.faster_rcnn import FasterRCNN
from tfcv.detection.runtime.evaluation import evaluate
from tfcv.detection.runtime.learning_rate import PiecewiseConstantWithWarmupSchedule
from tfcv.detection.trainers.faster_rcnn import FasterRCNNTrainer

def setup():
    if cfg.xla:
        os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit'
        logging.info('XLA is activated')

    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    if cfg.amp:
        policy = tf.keras.mixed_precision.experimental.Policy("mixed_float16", loss_scale="dynamic")
        tf.keras.mixed_precision.experimental.set_policy(policy)
        logging.info('AMP is activated')

def ctl_train():
    setup()
    dataset = Dataset()
    strategy = tf.distribute.MirroredStrategy()
    cfg.replicas = strategy.num_replicas_in_sync
    cfg.global_train_batch_size = cfg.train_batch_size * cfg.replicas
    cfg.global_eval_batch_size = cfg.eval_batch_size * cfg.replicas
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
        initialize(checkpoint)

        metrics = create_metrics()
        trainer = create_trainer(model, optimizer, metrics)

        dist_train_dataset = strategy.experimental_distribute_dataset(train_data)
        dist_eval_dataset = strategy.experimental_distribute_dataset(eval_data)
        trainer.compile()
        train_iter = iter(dist_train_dataset)

        total_steps = int(cfg.solver.epochs * dataset.train_size / cfg.global_train_batch_size)
        steps_per_epoch = int(total_steps * cfg.solver.evaluate_interval / cfg.solver.epochs)
        current_step = optimizer.iterations.numpy()

        for i in np.arange(0, cfg.solver.epochs, cfg.solver.evaluate_interval):
            i = np.round(i, 3)
            if i * steps_per_epoch < current_step:
                continue
            step_to_train = min(steps_per_epoch, total_steps - i * steps_per_epoch)
            trainer.train(step_to_train, train_iter)
            eval_results = trainer.evaluate(dist_eval_dataset)
            ckpt_path = os.path.join(cfg.model_dir, cfg.checkpoint_subdir, 'ckpt')
            checkpoint.save(ckpt_path)
            with open(f'res{i+1}.yaml', 'w') as fp:
                yaml.dump(eval_results, fp, Dumper=yaml.CDumper)



def create_model():
    if cfg.meta_arch == 'faster_rcnn':
        model = FasterRCNN()

    return model

def initialize(checkpoint):
    ckpt_path = os.path.join(cfg.model_dir, cfg.checkpoint_subdir)
    checkpoint_path = tf.train.latest_checkpoint(ckpt_path)
    if checkpoint_path is None:
        logging.info(f"No checkpoint was found in: {ckpt_path}")
        return
    checkpoint.restore(checkpoint_path).assert_consumed()
    logging.info(f"Loaded weights from checkpoint: {checkpoint_path}")

def create_metrics():

    if cfg.meta_arch == 'faster_rcnn':
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

def create_trainer(model, optimizer, metrics):
    if cfg.meta_arch == 'faster_rcnn':
        return FasterRCNNTrainer(cfg, model, optimizer, metrics)