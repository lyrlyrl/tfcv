import argparse

PARSER = argparse.ArgumentParser(
    description='custom implementation of cv models for TensorFlow 2.x',
    add_help=True)

# store hyperparameters
RUNTIME_GROUP = PARSER.add_argument_group('Runtime')
HYPER_GROUP = PARSER.add_argument_group('Hyperparameters')

RUNTIME_GROUP.add_argument(
    'mode',
    type=str,
    metavar='MODE',
    choices=['train', 'eval'],
    help='run mode',
)

RUNTIME_GROUP.add_argument(
    '--config_file',
    type=str,
    default=None,
    help='config file',
    required=True
)

RUNTIME_GROUP.add_argument(
    '--model_dir',
    type=str,
    default=None,
    help='workspace dir',
    required=True
)

RUNTIME_GROUP.add_argument(
    '--task',
    type=str,
    choices=['detection', 'classification'],
    required=True
)

HYPER_GROUP.add_argument(
    '--train_batch_size',
    type=int,
    default=4,
    metavar='N',
    help='Batch size (per GPU) used during training'
)

HYPER_GROUP.add_argument(
    '--eval_batch_size',
    type=int,
    default=4,
    metavar='N',
    help='Batch size used during evaluation'
)

HYPER_GROUP.add_argument(
    '--seed',
    type=int,
    default=None,
    metavar='SEED',
    help='Set a constant seed for reproducibility'
)

HYPER_GROUP.add_argument(
    '--xla',
    action='store_true',
    help='Enable XLA JIT Compiler'
)

HYPER_GROUP.add_argument(
    '--amp',
    action='store_true',
    help='Enable automatic mixed precision'
)

HYPER_GROUP.add_argument(
    '--strict_config',
    action='store_true',
    help='whether to use config hyperparameter'
)

# about train
HYPER_GROUP.add_argument(
    '--epochs',
    type=int,
    default=12,
    help='Number of training epochs'
)

HYPER_GROUP.add_argument(
    '--steps_per_loop',
    type=int,
    default=100,
    help='Number of steps per train loop'
)

HYPER_GROUP.add_argument(
    '--eval_samples',
    type=int,
    default=None,
    metavar='N',
    help='Number of evaluation samples'
)
