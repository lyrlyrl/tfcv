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
    '--num_gpus',
    type=int,
    default=1,
    help='number of gpus to use'
)

RUNTIME_GROUP.add_argument(
    '--task',
    type=str,
    choices=['detection', 'classification'],
    required=True
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

# about train
HYPER_GROUP.add_argument(
    '--steps_per_loop',
    type=int,
    default=100,
    metavar='N',
    help='Number of steps per train loop'
)

HYPER_GROUP.add_argument(
    '--config_override',
    help='A list of KEY=VALUE to overwrite those defined in config.yaml',
    nargs='+'
)
