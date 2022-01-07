import os
import argparse
import math
import tensorflow as tf
import shutil
from tfcv.config import update_cfg, config as cfg

from tfcv.datasets.coco.dataset import Dataset
from tfcv.utils.lazy_import import LazyImport
from tfcv.schedules.learning_rate import PiecewiseConstantWithWarmupSchedule

hvd = LazyImport('horovod.tensorflow')

PARSER = argparse.ArgumentParser(
    description='as child process'
)
PARSER.add_argument(
    '--model_dir',
    type=str,
    required=True
)
PARSER.add_argument(
    '--run_id',
    type=int,
    required=True
)
PARSER.add_argument(
    '--multiple_gpus',
    action='store_true',
    help='if train with multiple gpus'
)

def train(run_id, mg=False):
    dataset = Dataset()
    train_data = dataset.train_fn(
        cfg.train_batch_size, 
        shard=(hvd.size(), hvd.local_rank()) if mg else None)

    learning_rate = PiecewiseConstantWithWarmupSchedule(
        init_value=cfg.optimization.init_learning_rate,
        # scale boundaries from epochs to steps
        boundaries=[
            int(b * dataset.train_size / cfg.global_train_batch_size)
            for b in cfg.optimization.learning_rate_boundaries
        ],
        values=cfg.optimization.learning_rate_values,
        scale=cfg.global_train_batch_size
    )

    optimizer = tf.keras.optimizers.SGD(
        learning_rate=learning_rate,
        momentum=cfg.optimization.momentum
    )

    model = create_model()
    model.build(True)
    initialize_model(model, run_id)

    hooks = []
    trainer = create_trainer(model, optimizer, hooks)
    trainer.compile()
    train_iter = iter(train_data)
    total_steps = math.ceil(cfg.solver.epochs * dataset.train_size / cfg.global_train_batch_size)
    trainer.train(total_steps, train_iter)

    step = optimizer.iterations.numpy()
    model.save_weights(
        os.path.join(cfg.model_dir, 'model_weights', str(run_id), f'{model.name}-{str(step)}.npz')
    )


def create_model():
    pass

def create_trainer(model, optimizer, hooks=[]):
    pass

def initialize_model(model, run_id):
    all_weight_dir = os.path.join(cfg.model_dir, 'model_weights')
    historys = list(map(int, os.listdir(all_weight_dir)))
    if max(historys) >= run_id:
        for i in historys:
            if i >= run_id:
                shutil.rmtree(os.path.join())
    weight_dir = os.path.join(all_weight_dir, str(run_id))
    if os.path.isdir(all_weight_dir):
        model_name = model.name
        weights = [name for name in os.listdir(weight_dir) if name.endswith('.npz')]
    else:
        os.makedirs(weight_dir)
    

if __name__ == '__main__':
    arguments = PARSER.parse_args()

    if arguments.multiple_gpus:
        hvd.init()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus and arguments.multiple_gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
    
    model_dir = arguments.model_dir
    model_dir = os.path.abspath(model_dir)
    config_path = os.path.join(model_dir, 'train_config.yaml')
    params = update_cfg(config_path)
    cfg.from_dict(params)
    cfg.model_dir = model_dir
    if arguments.multiple_gpus:
        cfg.global_train_batch_size = cfg.train_batch_size * hvd.size()
    else:
        cfg.global_train_batch_size = cfg.train_batch_size
    cfg.freeze()
    
    train(arguments.run_id, arguments.multiple_gpus)

