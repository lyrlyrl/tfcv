import tensorflow as tf
import os
import glob
import logging

from tfcv.distribute import MPI_is_distributed, MPI_local_rank, MPI_size

TRAIN_SPLIT_PATTERN = 'train*.tfrecord'
EVAL_SPLIT_PATTERN = 'val*.tfrecord'

class TFDataset:

    def __init__(self, params):
        self._params = params
        self._logger = logging.getLogger('dataset')

    def train(self, parser, batch_size):
        data_dir = os.path.expanduser(os.path.join(self._params.data.dir, 'tfrecords'))
        train_files = glob.glob(os.path.join(data_dir, TRAIN_SPLIT_PATTERN))

        data = tf.data.TFRecordDataset(train_files)
        if MPI_is_distributed():
            data = data.shard(MPI_size(), MPI_local_rank())

        data = data.cache()
        data = data.shuffle(buffer_size=4096, reshuffle_each_iteration=True, seed=self._params.seed)
        data = data.repeat()

        data = data.map(
            parser,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        data = data.batch(batch_size=batch_size, drop_remainder=True)
        
        data = data.prefetch(buffer_size=tf.data.AUTOTUNE)

        return data
    
    def eval(self, parser, batch_size):
        data_dir = os.path.expanduser(os.path.join(self._params.data.dir, 'tfrecords'))
        eval_files = glob.glob(os.path.join(data_dir, EVAL_SPLIT_PATTERN))
        data = tf.data.TFRecordDataset(eval_files)

        if MPI_is_distributed():
            data = data.shard(MPI_size(), MPI_local_rank())
        
        data = data.cache()
        data = data.map(
            parser,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        data = data.batch(batch_size=batch_size, drop_remainder=True)

        data = data.prefetch(buffer_size=tf.data.AUTOTUNE)
