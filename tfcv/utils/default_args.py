import argparse

PARSER = argparse.ArgumentParser(
    description='custom implementation of cv models for TensorFlow 2.x',
    add_help=False)

# store hyperparameters
HYPER_GROUP = PARSER.add_argument_group('Hyperparameters')
SOLVER_GROUP = PARSER.add_argument_group('Solver')

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

# about train
SOLVER_GROUP.add_argument(
    '--epochs',
    type=int,
    default=12,
    help='Number of training epochs'
)

SOLVER_GROUP.add_argument(
    '--steps_per_loop',
    type=int,
    default=100,
    help='Number of steps per train loop'
)

SOLVER_GROUP.add_argument(
    '--eval_samples',
    type=int,
    default=None,
    metavar='N',
    help='Number of evaluation samples'
)