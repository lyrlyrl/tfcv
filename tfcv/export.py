import argparse

PARSER = argparse.ArgumentParser(
    description='export pretreaned models',
    add_help=True)

PARSER.add_argument(
    '--model_dir',
    type=str,
    default=None,
    help='workspace dir',
    required=True
)

PARSER.add_argument(
    '--ckpt_number',
    type=int,
    help='checkpoint number to export',
    required=True
)

PARSER.add_argument(
    '--precision',
    type=str,
    default='fp32',
    choices=['fp32', 'fp16', 'int8'],
    help='workspace dir',
)

if __name__ == '__main__':
    pass