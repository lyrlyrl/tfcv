import logging
import os
from argparse import Namespace

import dllogger

import tfcv

from tfcv.detection.runtime.run import run_training, run_inference, run_evaluation, ctl_training
from tfcv.detection.utils.dllogger import LoggingBackend

from tfcv.detection.arguments import PARSER
from tfcv.detection.config import CONFIG
from tfcv.detection.dataset.dataset import Dataset

if __name__ == '__main__':
    # setup params
    arguments = PARSER.parse_args()
    params = Namespace(**{**vars(CONFIG), **vars(arguments)})

    # setup logging
    # noinspection PyArgumentList
    logging.basicConfig(
        # level=logging.DEBUG if params.verbose else logging.INFO,
        level=logging.DEBUG,
        format='{asctime} {levelname:.1} {name:15} {message}',
        style='{'
    )

    # remove custom tf handler that logs to stderr
    logging.getLogger('tensorflow').setLevel(logging.DEBUG)
    logging.getLogger('tensorflow').handlers.clear()

    # setup dllogger
    dllogger.init(backends=[
        dllogger.JSONStreamBackend(verbosity=dllogger.Verbosity.VERBOSE, filename=params.log_file),
        LoggingBackend(verbosity=dllogger.Verbosity.VERBOSE)
    ])
    dllogger.log(step='PARAMETER', data=vars(params))

    # setup dataset
    dataset = Dataset(params)

    ctl_training(dataset, params)