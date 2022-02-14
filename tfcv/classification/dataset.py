import tensorflow as tf

import nvidia.dali.fn as fn
import nvidia.dali.types as types
import nvidia.dali.tfrecord as tfrec
import nvidia.dali.plugin.tf as dali_tf
from nvidia.dali.pipeline import pipeline_def

from tfcv.distribute import MPI_local_rank, MPI_size, MPI_rank

TRAIN_SPLIT_PATTERN = 'train-*'
EVAL_SPLIT_PATTERN = 'val-*'

@pipeline_def
def get_dali_pipeline(
        tfrec_filenames,
        tfrec_idx_filenames,
        height, width,
        shard_id,
        num_gpus,
        dali_cpu=True,
        training=True):

    inputs = fn.readers.tfrecord(
            path=tfrec_filenames,
            index_path=tfrec_idx_filenames,
            random_shuffle=training,
            shard_id=shard_id,
            num_shards=num_gpus,
            initial_fill=10000,
            features={
                'image/encoded': tfrec.FixedLenFeature((), tfrec.string, ""),
                'image/class/label': tfrec.FixedLenFeature([1], tfrec.int64,  -1),
                'image/class/text': tfrec.FixedLenFeature([ ], tfrec.string, ''),
                'image/object/bbox/xmin': tfrec.VarLenFeature(tfrec.float32, 0.0),
                'image/object/bbox/ymin': tfrec.VarLenFeature(tfrec.float32, 0.0),
                'image/object/bbox/xmax': tfrec.VarLenFeature(tfrec.float32, 0.0),
                'image/object/bbox/ymax': tfrec.VarLenFeature(tfrec.float32, 0.0)})

    decode_device = "cpu" if dali_cpu else "mixed"
    resize_device = "cpu" if dali_cpu else "gpu"
    if training:
        images = fn.decoders.image_random_crop(
            inputs["image/encoded"],
            device=decode_device,
            output_type=types.RGB,
            random_aspect_ratio=[0.75, 1.25],
            random_area=[0.05, 1.0],
            num_attempts=100,
            # ask HW NVJPEG to allocate memory ahead for the biggest image in the data set to avoid reallocations in runtime
            preallocate_width_hint=5980 if decode_device == 'mixed' else 0,
            preallocate_height_hint=6430 if decode_device == 'mixed' else 0)
        images = fn.resize(images, device=resize_device, resize_x=width, resize_y=height)
    else:
        images = fn.decoders.image(
            inputs["image/encoded"],
            device=decode_device,
            output_type=types.RGB)
        # Make sure that every image > 224 for CropMirrorNormalize
        images = fn.resize(images, device=resize_device, resize_shorter=256)

    images = fn.crop_mirror_normalize(
        images.gpu(),
        dtype=types.FLOAT,
        crop=(height, width),
        mean=[123.68, 116.78, 103.94],
        std=[58.4, 57.12, 57.3],
        output_layout="HWC",
        mirror = fn.random.coin_flip())
    labels = inputs["image/class/label"].gpu()

    labels -= 1 # Change to 0-based (don't use background class)
    return images, labels

class DALIPreprocessor(object):
    def __init__(self,
                filenames,
                idx_filenames,
                height, width,
                batch_size,
                num_threads,
                dtype=tf.uint8,
                dali_cpu=False,
                deterministic=False,
                training=False):
        device_id = MPI_local_rank()
        shard_id = MPI_rank()
        num_gpus = MPI_size()
        self.pipe = get_dali_pipeline(
            tfrec_filenames=filenames,
            tfrec_idx_filenames=idx_filenames,
            height=height,
            width=width,
            batch_size=batch_size,
            num_threads=num_threads,
            device_id=device_id,
            shard_id=shard_id,
            num_gpus=num_gpus,
            dali_cpu=dali_cpu,
            training=training,
            seed=7 * (1 + MPI_rank()) if deterministic else None)

        self.daliop = dali_tf.DALIIterator()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.device_id = device_id

        self.dalidataset = dali_tf.DALIDataset(
            pipeline=self.pipe,
            output_shapes=((batch_size, height, width, 3), (batch_size)),
            batch_size=batch_size,
            output_dtypes=(tf.float32, tf.int64),
            device_id=device_id)

    def get_device_minibatches(self):
        with tf.device("/gpu:0"):
            images, labels = self.daliop(
                pipeline=self.pipe,
                shapes=[(self.batch_size, self.height, self.width, 3), ()],
                dtypes=[tf.float32, tf.int64],
                device_id=self.device_id)
        return images, labels

    def get_device_dataset(self):
        return self.dalidataset

