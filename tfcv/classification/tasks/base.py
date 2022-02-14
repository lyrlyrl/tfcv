import tensorflow as tf

from tfcv.core import Task
from tfcv.distribute import MPI_local_rank, MPI_size, MPI_rank
from tfcv.common import expand_image_shape
from tfcv.ops import preprocess_ops

from tfcv.classification.models.resnet import ResNet


class ClassificationTask(Task):

    def __init__(self, params):
        self._params = params

    def create_model(self):
        if self._params.meta_arch == 'resnet':
            return ResNet(
                self._params.model_id,
                expand_image_shape(self._params.data.image_size),
                num_classes=self._params.num_classes
            )

    def train_forward(self, model, inputs):
        images, labels = inputs
        predictions = model(images, training=True)
        if self._params.losses.one_hot:
            loss_func = tf.keras.losses.CategoricalCrossentropy(
                label_smoothing=self._params.losses.label_smoothing)
        else:
            loss_func = tf.keras.losses.SparseCategoricalCrossentropy()
        losses = {'model_loss': loss_func(labels, predictions)}
        losses['l2_regularization_loss'] = tf.add_n([
            tf.nn.l2_loss(tf.cast(v, dtype=tf.float32))
            for v in model.trainable_variables
            if not any([pattern in v.name for pattern in ["batch_normalization", "bias", "beta"]])
        ]) * self._params.loss.l2_weight_decay
        raw_loss = tf.math.reduce_sum(list(losses.values()))

        to_update = {
            'train_top1': [labels, predictions],
            'train_top5': [labels, predictions]
        }

        return (raw_loss, losses, to_update)

    def validate_forward(self, model, inputs):
        images, labels = inputs
        predictions = model(images, training=False)
        loss_func = tf.keras.losses.SparseCategoricalCrossentropy()
        losses = {'model_loss': loss_func(labels, predictions)}

        raw_loss = tf.math.reduce_sum(list(losses.values()))
        
        to_update = {
            'val_top1': [labels, predictions],
            'val_top5': [labels, predictions]
        }
        return (raw_loss, losses, to_update)

    def inference_forward(self, model, inputs):
        predictions = model.call(inputs, training=False)
        return predictions

    def inference_preprocess(self, image):
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = preprocess_ops.normalize_image(image, self._params.data.pixel_std, self._params.data.pixel_mean)