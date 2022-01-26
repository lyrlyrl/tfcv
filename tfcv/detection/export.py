import logging
import argparse
import os
import yaml

import tensorflow as tf

from tfcv import Predictor
from tfcv.config import update_cfg, config as cfg
from tfcv.detection.train import setup, create_task

PARSER = argparse.ArgumentParser(
    description='custom implementation of object detection models for TensorFlow 2.x',
    add_help=True)

PARSER.add_argument(
    '--image_size',
    type=str,
    help='Set the input shape of the graph, as comma-separated dimensions in HW format,'
        'eg: 720,1280',
    required=True
)

PARSER.add_argument(
    '--batch_size',
    type=int,
    help='Set the input batch_size of the graph,'
        'eg: 1',
    default=1
)

PARSER.add_argument(
    '--input_format',
    type=str,
    help='Set the input format of the graph, '
        'eg: NCHW,NHWC',
    default='NHWC',
    choices=['NCHW', 'NHWC']
)

PARSER.add_argument(
    '--config_file',
    type=str,
    help='workspace dir',
    required=True
)

PARSER.add_argument(
    '--checkpoint',
    type=str,
    required=True,
    help='workspace dir',
)

PARSER.add_argument(
    '--savedmodel_dir',
    type=str,
    required=True,
    help='export model to tf saved_model format'
)

PARSER.add_argument(
    '--onnx_dir',
    type=str,
    default=None,
    help='export model to onnx format'
)

def export(ckpt, savedmodel_dir, onnx_dir=None):
    task = create_task(cfg)
    model = task.create_model()
    checkpoint = tf.train.Checkpoint(model=model)

    checkpoint.restore(ckpt).expect_partial()

    predictor = Predictor(cfg, model, task)

    input_shape = [cfg.export.batch_size]
    if cfg.export.input_format == 'NCHW':
        input_shape = input_shape + [3] + list(cfg.export.image_size)
    else:
        input_shape = input_shape + list(cfg.export.image_size) + [3]

    tf.saved_model.save(
        predictor,
        savedmodel_dir,
        signatures=predictor.service_step.get_concrete_function(
            tf.TensorSpec(input_shape, tf.uint8)
        )
    )

if __name__ == '__main__':
    arguments = PARSER.parse_args()

    params = update_cfg(arguments.config_file)
    cfg.from_dict(params)
    cfg.export.batch_size = arguments.batch_size
    cfg.export.input_format = arguments.input_format

    image_size = arguments.image_size.split(",")
    assert len(image_size) == 2
    for i in range(len(image_size)):
        image_size[i] = int(image_size[i])
        assert image_size[i] >= 1
    cfg.export.image_size = image_size

    setup(cfg)

    export(arguments.checkpoint, arguments.savedmodel_dir, arguments.onnx_dir)