import argparse

RUN_PARSER = argparse.ArgumentParser(
    description='custom implementation of cv models for TensorFlow 2.x',
    add_help=True)

# store hyperparameters
COMMON_GROUP = RUN_PARSER.add_argument_group('Common')
TRAIN_GROUP = RUN_PARSER.add_argument_group('Train')
EVAL_GROUP = RUN_PARSER.add_argument_group('Eval')
EXPORT_GROUP = RUN_PARSER.add_argument_group('Export')

COMMON_GROUP.add_argument(
    'mode',
    type=str,
    metavar='MODE',
    choices=['train', 'eval', 'export'],
    help='run mode',
)

COMMON_GROUP.add_argument(
    '--config_file',
    type=str,
    default=None,
    help='config file',
    required=True
)

COMMON_GROUP.add_argument(
    '--model_dir',
    type=str,
    default=None,
    help='workspace dir',
    required=True
)

COMMON_GROUP.add_argument(
    '--gpu_ids',
    type=int,
    nargs='+',
    help='id of gpus to use'
)

COMMON_GROUP.add_argument(
    '--seed',
    type=int,
    default=None,
    metavar='SEED',
    help='Set a constant seed for reproducibility'
)

COMMON_GROUP.add_argument(
    '--noxla',
    action='store_true',
    help='Disable XLA JIT Compiler'
)

COMMON_GROUP.add_argument(
    '--config_override',
    help='A list of KEY=VALUE to overwrite those defined in config.yaml',
    nargs='+'
)

# about train
TRAIN_GROUP.add_argument(
    '--steps_per_loop',
    type=int,
    default=100,
    metavar='N',
    help='Number of steps per train loop'
)

TRAIN_GROUP.add_argument(
    '--amp',
    action='store_true',
    help='Enable automatic mixed precision'
)

# about evaluate
EVAL_GROUP.add_argument(
    '--eval_number',
    type=int,
    default=None,
    nargs='+',
    help='checkpoint number to evaluate',
)

# about export
EXPORT_GROUP.add_argument(
    '--export_to_savedmodel',
    action='store_true',
    help='export model to tf saved_model format'
)