import tensorflow as tf


class RPNHead(tf.keras.layers.Layer):

    def __init__(self, 
                name,
                num_anchors,
                num_filters,
                *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)
        """Shared RPN heads."""

        # TODO(chiachenc): check the channel depth of the first convolution.
        self._conv = tf.keras.layers.Conv2D(
            num_filters,
            kernel_size=3,
            strides=1,
            activation=tf.nn.relu,
            bias_initializer=tf.keras.initializers.Zeros(),
            kernel_initializer=tf.random_normal_initializer(stddev=0.01),
            padding='same',
            name='rpn-conv'
        )

        # Proposal classification scores
        # scores = tf.keras.layers.Conv2D(
        self._classifier = tf.keras.layers.Conv2D(
            num_anchors,
            kernel_size=1,
            strides=1,
            bias_initializer=tf.keras.initializers.Zeros(),
            kernel_initializer=tf.random_normal_initializer(stddev=0.01),
            padding='valid',
            name='rpn-score'
        )

        # Proposal bbox regression deltas
        # bboxes = tf.keras.layers.Conv2D(
        self._box_regressor = tf.keras.layers.Conv2D(
            4 * num_anchors,
            kernel_size=1,
            strides=1,
            bias_initializer=tf.keras.initializers.Zeros(),
            kernel_initializer=tf.random_normal_initializer(stddev=0.01),
            padding='valid',
            name='rpn-box'
        )

    def call(self, inputs, *args, **kwargs):
        net = self._conv(inputs)
        scores = self._classifier(net)
        bboxes = self._box_regressor(net)

        return scores, bboxes


class BoxHead(tf.keras.layers.Layer):

    def __init__(self, num_classes=91, mlp_head_dim=1024, *args, **kwargs):
        """Box and class branches for the Mask-RCNN model.

        Args:
        roi_features: A ROI feature tensor of shape
          [batch_size, num_rois, height_l, width_l, num_filters].
        num_classes: a integer for the number of classes.
        mlp_head_dim: a integer that is the hidden dimension in the fully-connected
          layers.
        """
        super().__init__(*args, **kwargs)

        self._num_classes = num_classes
        self._mlp_head_dim = mlp_head_dim

        self._dense_fc6 = tf.keras.layers.Dense(
            units=mlp_head_dim,
            activation=tf.nn.relu,
            name='fc6'
        )

        self._dense_fc7 = tf.keras.layers.Dense(
            units=mlp_head_dim,
            activation=tf.nn.relu,
            name='fc7'
        )

        self._dense_class = tf.keras.layers.Dense(
            num_classes,
            kernel_initializer=tf.random_normal_initializer(stddev=0.01),
            bias_initializer=tf.keras.initializers.Zeros(),
            name='class-predict'
        )

        self._dense_box = tf.keras.layers.Dense(
            num_classes * 4,
            kernel_initializer=tf.random_normal_initializer(stddev=0.001),
            bias_initializer=tf.keras.initializers.Zeros(),
            name='box-predict'
        )

    def call(self, inputs, **kwargs):
        """
        Returns:
        class_outputs: a tensor with a shape of
          [batch_size, num_rois, num_classes], representing the class predictions.
        box_outputs: a tensor with a shape of
          [batch_size, num_rois, num_classes * 4], representing the box predictions.
        box_features: a tensor with a shape of
          [batch_size, num_rois, mlp_head_dim], representing the box features.
        """

        # reshape inputs before FC.
        batch_size, num_rois, height, width, filters = inputs.get_shape().as_list()
        
        net = tf.reshape(inputs, [batch_size, num_rois, height * width * filters])

        net = self._dense_fc6(net)

        box_features = self._dense_fc7(net)

        class_outputs = self._dense_class(box_features)

        box_outputs = self._dense_box(box_features)

        return class_outputs, box_outputs, box_features


class MaskHead(tf.keras.layers.Layer):

    @staticmethod
    def _get_stddev_equivalent_to_msra_fill(kernel_size, fan_out):
        """Returns the stddev of random normal initialization as MSRAFill."""
        # Reference: https://github.com/pytorch/pytorch/blob/master/caffe2/operators/filler_op.h#L445-L463
        # For example, kernel size is (3, 3) and fan out is 256, stddev is 0.029.
        # stddev = (2/(3*3*256))^0.5 = 0.029
        return (2 / (kernel_size[0] * kernel_size[1] * fan_out)) ** 0.5

    def __init__(
            self,
            num_classes=91,
            mrcnn_resolution=28,
            *args,
            **kwargs
    ):
        """Mask branch for the Mask-RCNN model.

        Args:
        roi_features: A ROI feature tensor of shape
          [batch_size, num_rois, height_l, width_l, num_filters].
        num_classes: an integer for the number of classes.
        mrcnn_resolution: an integer that is the resolution of masks.
        """
        super().__init__(*args, **kwargs)

        self._num_classes = num_classes
        self._mrcnn_resolution = mrcnn_resolution

        self._conv_stage1 = list()
        kernel_size = (3, 3)
        fan_out = 256

        init_stddev = MaskHead._get_stddev_equivalent_to_msra_fill(kernel_size, fan_out)

        for conv_id in range(4):
            self._conv_stage1.append(tf.keras.layers.Conv2D(
                fan_out,
                kernel_size=kernel_size,
                strides=(1, 1),
                padding='same',
                dilation_rate=(1, 1),
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(stddev=init_stddev),
                bias_initializer=tf.keras.initializers.Zeros(),
                name='mask-conv-l%d' % conv_id
            ))

        kernel_size = (2, 2)
        fan_out = 256

        init_stddev = MaskHead._get_stddev_equivalent_to_msra_fill(kernel_size, fan_out)

        self._conv_stage2 = tf.keras.layers.Conv2DTranspose(
            fan_out,
            kernel_size=kernel_size,
            strides=(2, 2),
            padding='valid',
            activation=tf.nn.relu,
            kernel_initializer=tf.random_normal_initializer(stddev=init_stddev),
            bias_initializer=tf.keras.initializers.Zeros(),
            name='conv5-mask'
        )

        kernel_size = (1, 1)
        fan_out = self._num_classes

        init_stddev = MaskHead._get_stddev_equivalent_to_msra_fill(kernel_size, fan_out)

        self._conv_stage3 = tf.keras.layers.Conv2D(
            fan_out,
            kernel_size=kernel_size,
            strides=(1, 1),
            padding='valid',
            kernel_initializer=tf.random_normal_initializer(stddev=init_stddev),
            bias_initializer=tf.keras.initializers.Zeros(),
            name='mask_fcn_logits'
        )

    def call(self, inputs, **kwargs):
        """
        Args:
            inputs: tuple of two tensors:
                mask_roi_features: a Tensor of shape:
                  [batch_size, num_boxes, output_size, output_size, num_filters].
                class_indices: a Tensor of shape [batch_size, num_rois], indicating
                  which class the ROI is.
            training: whether to build the model for training (or inference).
        Returns:
            mask_outputs: a tensor with a shape of
              [batch_size, num_masks, mask_height, mask_width],
              representing the mask predictions.
            fg_gather_indices: a tensor with a shape of [batch_size, num_masks, 2],
              representing the fg mask targets.
        Raises:
            ValueError: If boxes is not a rank-3 tensor or the last dimension of
              boxes is not 4.
        """
        mask_roi_features, class_indices = inputs
        indices_dtype = tf.int32
        # fixed problems when running with Keras AMP
        class_indices = tf.cast(class_indices, dtype=indices_dtype)

        batch_size, num_rois, height, width, filters = mask_roi_features.get_shape().as_list()

        net = tf.reshape(mask_roi_features, [-1, height, width, filters])

        for conv_id in range(4):
            net = self._conv_stage1[conv_id](net)

        net = self._conv_stage2(net)

        mask_outputs = self._conv_stage3(net)

        mask_outputs = tf.reshape(
            mask_outputs,
            [-1, num_rois, self._mrcnn_resolution, self._mrcnn_resolution, self._num_classes]
        )

        with tf.name_scope('masks_post_processing'):

            mask_outputs = tf.transpose(a=mask_outputs, perm=[0, 1, 4, 2, 3])

            if batch_size == 1:
                indices = tf.reshape(
                    tf.reshape(
                        tf.range(num_rois, dtype=indices_dtype),
                        [batch_size, num_rois, 1]
                    ) * self._num_classes + tf.expand_dims(class_indices, axis=-1),
                    [batch_size, -1]
                )

                mask_outputs = tf.gather(
                    tf.reshape(mask_outputs,
                               [batch_size, -1, self._mrcnn_resolution, self._mrcnn_resolution]),
                    indices,
                    axis=1
                )

                mask_outputs = tf.squeeze(mask_outputs, axis=1)
                mask_outputs = tf.reshape(
                    mask_outputs,
                    [batch_size, num_rois, self._mrcnn_resolution, self._mrcnn_resolution])

            else:
                batch_indices = (
                        tf.expand_dims(tf.range(batch_size, dtype=indices_dtype), axis=1) *
                        tf.ones([1, num_rois], dtype=indices_dtype)
                )

                mask_indices = (
                        tf.expand_dims(tf.range(num_rois, dtype=indices_dtype), axis=0) *
                        tf.ones([batch_size, 1], dtype=indices_dtype)
                )

                gather_indices = tf.stack([batch_indices, mask_indices, class_indices], axis=2)

                mask_outputs = tf.gather_nd(mask_outputs, gather_indices)

        return mask_outputs
