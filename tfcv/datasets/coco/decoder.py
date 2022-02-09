import tensorflow as tf

def _get_source_id_from_encoded_image(parsed_tensors):
    return tf.strings.as_string(tf.strings.to_hash_bucket_fast(parsed_tensors['image/encoded'], 2 ** 63 - 1))

class COCOExampleDecoder:
    """Tensorflow Example proto decoder."""

    def __init__(self, use_instance_mask=False, include_source_id=False):
        self._include_mask = use_instance_mask
        self._include_source_id = include_source_id
        self._keys_to_features = {
            'image/encoded': tf.io.FixedLenFeature((), tf.string),
            'image/source_id': tf.io.FixedLenFeature((), tf.string),
            'image/height': tf.io.FixedLenFeature((), tf.int64),
            'image/width': tf.io.FixedLenFeature((), tf.int64),
            'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
            'image/object/class/label': tf.io.VarLenFeature(tf.int64),
            'image/object/area': tf.io.VarLenFeature(tf.float32),
            'image/object/is_crowd': tf.io.VarLenFeature(tf.int64),
        }
        if use_instance_mask:
            self._keys_to_features.update({
                'image/object/mask': tf.io.VarLenFeature(tf.string),
            })

    def _decode_image(self, parsed_tensors):
        """Decodes the image and set its static shape."""
        image = tf.io.decode_image(parsed_tensors['image/encoded'], channels=3)
        image.set_shape([None, None, 3])
        return image

    def _decode_boxes(self, parsed_tensors):
        """Concat box coordinates in the format of [ymin, xmin, ymax, xmax]."""
        xmin = parsed_tensors['image/object/bbox/xmin']
        xmax = parsed_tensors['image/object/bbox/xmax']
        ymin = parsed_tensors['image/object/bbox/ymin']
        ymax = parsed_tensors['image/object/bbox/ymax']
        return tf.stack([ymin, xmin, ymax, xmax], axis=-1)

    def _decode_masks(self, parsed_tensors):
        """Decode a set of PNG masks to the tf.float32 tensors."""

        def _decode_png_mask(png_bytes):
            mask = tf.squeeze(tf.io.decode_png(png_bytes, channels=1, dtype=tf.uint8), axis=-1)
            mask = tf.cast(mask, dtype=tf.float32)
            mask.set_shape([None, None])
            return mask

        height = parsed_tensors['image/height']
        width = parsed_tensors['image/width']
        masks = parsed_tensors['image/object/mask']
        return tf.cond(
            pred=tf.greater(tf.size(input=masks), 0),
            true_fn=lambda: tf.map_fn(_decode_png_mask, masks, dtype=tf.float32),
            false_fn=lambda: tf.zeros([0, height, width], dtype=tf.float32)
        )

    def decode(self, serialized_example):
        """Decode the serialized example.

    Args:
      serialized_example: a single serialized tf.Example string.

    Returns:
      decoded_tensors: a dictionary of tensors with the following fields:
        - image: a uint8 tensor of shape [None, None, 3].
        - source_id: a string scalar tensor.
        - height: an integer scalar tensor.
        - width: an integer scalar tensor.
        - groundtruth_classes: a int64 tensor of shape [None].
        - groundtruth_is_crowd: a bool tensor of shape [None].
        - groundtruth_area: a float32 tensor of shape [None].
        - groundtruth_boxes: a float32 tensor of shape [None, 4].
        - groundtruth_instance_masks: a float32 tensor of shape
            [None, None, None].
        - groundtruth_instance_masks_png: a string tensor of shape [None].
    """
        parsed_tensors = tf.io.parse_single_example(
            serialized=serialized_example, features=self._keys_to_features)
        for k in parsed_tensors:
            if isinstance(parsed_tensors[k], tf.SparseTensor):
                if parsed_tensors[k].dtype == tf.string:
                    parsed_tensors[k] = tf.sparse.to_dense(
                        parsed_tensors[k], default_value='')
                else:
                    parsed_tensors[k] = tf.sparse.to_dense(
                        parsed_tensors[k], default_value=0)

        image = self._decode_image(parsed_tensors)
        boxes = self._decode_boxes(parsed_tensors)
        is_crowd = tf.cast(parsed_tensors['image/object/is_crowd'], dtype=tf.bool)

        if self._include_mask:
            masks = self._decode_masks(parsed_tensors)

        decoded_tensors = {
            'image': image,
            'height': parsed_tensors['image/height'],
            'width': parsed_tensors['image/width'],
            'groundtruth_classes': parsed_tensors['image/object/class/label'],
            'groundtruth_is_crowd': is_crowd,
            'groundtruth_area': parsed_tensors['image/object/area'],
            'groundtruth_boxes': boxes,
        }
        if self._include_source_id:
            source_id = tf.cond(
                pred=tf.greater(tf.strings.length(input=parsed_tensors['image/source_id']), 0),
                true_fn=lambda: parsed_tensors['image/source_id'],
                false_fn=lambda: _get_source_id_from_encoded_image(parsed_tensors)
            )
            decoded_tensors['source_id'] = source_id

        if self._include_mask:
            decoded_tensors.update({
                'groundtruth_instance_masks': masks,
                # 'mask_png_bytes': parsed_tensors['image/object/mask'],
            })

        return decoded_tensors