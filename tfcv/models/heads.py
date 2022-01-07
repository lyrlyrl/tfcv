import tensorflow as tf

from tfcv.layers.base import Layer

from tfcv.layers.conv2d import Conv2D, Conv2DTranspose
from tfcv.layers.linear import Linear
from tfcv.layers.utils import need_build, build_layers

class RPNHead(Layer):

    default_name = 'rpn_head'

    def __init__(
        self, 
        num_anchors: int,
        num_filters: int,
        trainable=True, name=None):
        self._init(locals())
        super(RPNHead, self).__init__(trainable=trainable, name=name)
        self._layers['rpn_conv'] = Conv2D(
            num_filters,
            3,
            1,
            padding='same',
            kernel_initializer=tf.random_normal_initializer(stddev=0.01),
            use_bias=True,
            bias_initializer=tf.keras.initializers.Zeros(),
            name = 'rpn_conv'
        )
        self._layers['rpn_score'] = Conv2D(
            num_anchors,
            1,
            1,
            padding='valid',
            kernel_initializer=tf.random_normal_initializer(stddev=0.01),
            use_bias=True,
            bias_initializer=tf.keras.initializers.Zeros(),
            name = 'rpn_score'
        )
        self._layers['rpn_box'] = Conv2D(
            num_anchors * 4,
            1,
            1,
            padding='valid',
            kernel_initializer=tf.random_normal_initializer(stddev=0.01),
            use_bias=True,
            bias_initializer=tf.keras.initializers.Zeros(),
            name = 'rpn_box'
        )
    
    def _build(self, input_shape):
        with tf.name_scope(self.name):
            self._layers['rpn_conv'].build(input_shape)
            output_specs = self._layers['rpn_conv'].output_specs
            self._layers['rpn_score'].build(output_specs)
            self._layers['rpn_box'].build(output_specs)
        self._output_specs = self.compute_output_specs(input_shape)
    
    def compute_output_specs(self, input_shape):
        internal = self._layers['rpn_conv'].compute_output_specs(input_shape)
        return (self._layers['rpn_score'].compute_output_specs(internal), 
                self._layers['rpn_box'].compute_output_specs(internal))
    
    def call(self, inputs, training=None):
        net = self._layers['rpn_conv'](inputs, training=training)
        net = tf.nn.relu(net)
        scores = self._layers['rpn_score'](net, training=training)
        bboxes = self._layers['rpn_box'](net, training=training)

        return (scores, bboxes)

class MultilevelRPNHead(RPNHead):

    def __init__(
        self, 
        min_level: int,
        max_level: int,
        num_anchors: int, 
        num_filters: int, 
        trainable=True, 
        name=None):
        self._init(locals())
        super(MultilevelRPNHead, self).__init__(num_anchors, num_filters, trainable=trainable, name=name)
    def compute_output_specs(self, input_shape):
        scores = dict()
        bboxes = dict()
        for level in range(self.min_level, self.max_level+1):
            scores[str(level)], bboxes[str(level)] = super(MultilevelRPNHead, self).compute_output_specs(input_shape[str(level)])
        return (scores, bboxes)
    def _build(self, input_shape):
        with tf.name_scope(self.name):
            self._layers['rpn_conv'].build(list(input_shape.values())[0])
            output_specs = self._layers['rpn_conv'].output_specs
            self._layers['rpn_score'].build(output_specs)
            self._layers['rpn_box'].build(output_specs)
        self._output_specs = self.compute_output_specs(input_shape)
    @need_build
    def call(self, inputs, training=None):
        scores = dict()
        bboxes = dict()
        for level in range(self.min_level, self.max_level+1):
            net = self._layers['rpn_conv'](inputs[str(level)], training=training)
            net = tf.nn.relu(net)
            scores[str(level)] = self._layers['rpn_score'](net, training=training)
            bboxes[str(level)] = self._layers['rpn_box'](net, training=training)
        return (scores, bboxes)

class FCBoxHead(Layer):

    default_name = 'fcbox_head'

    def __init__(
        self, 
        num_classes=91,
        mlp_head_dim=1024,
        trainable=True, 
        name=None):
        self._init(locals())
        super().__init__(trainable=trainable, name=name)
        self._layers['fc6'] = Linear(mlp_head_dim, trainable=trainable, name='fc6')
        self._layers['fc7'] = Linear(mlp_head_dim, trainable=trainable, name='fc7')
        self._layers['class_predict'] = Linear(
            num_classes, 
            kernel_initializer=tf.random_normal_initializer(stddev=0.01),
            trainable=trainable, name='class_predict')
        self._layers['box_predict'] = Linear(
            num_classes * 4, 
            kernel_initializer=tf.random_normal_initializer(stddev=0.001),
            trainable=trainable, name='box_predict')

    def _build(self, input_shape):
        batch_size, num_rois, height, width, filters = input_shape
        with tf.name_scope(self.name):
            self._layers['fc6'].build([batch_size, num_rois, height * width * filters])
            self._layers['fc7'].build(self._layers['fc6'].output_specs)
            self._layers['class_predict'].build(self._layers['fc7'].output_specs)
            self._layers['box_predict'].build(self._layers['fc7'].output_specs)
        self._output_specs = (self._layers['class_predict'].output_specs, 
            self._layers['box_predict'].output_specs)

    def compute_output_specs(self, input_shape):
        batch_size, num_rois, _, _, _ = input_shape
        return (
            [batch_size, num_rois, self.num_classes],
            [batch_size, num_rois, self.num_classes * 4]
        )

    def call(self, inputs, training=None):
        batch_size, num_rois, height, width, filters = inputs.get_shape().as_list()
        net = tf.reshape(inputs, [batch_size, num_rois, height * width * filters])

        net = self._layers['fc6'](net, training)
        net = tf.nn.relu(net)
        box_features = self._layers['fc7'](net, training)
        box_features = tf.nn.relu(box_features)

        class_outputs = self._layers['class_predict'](box_features, training)

        box_outputs = self._layers['box_predict'](box_features, training)

        return (class_outputs, box_outputs)

class MaskHead(Layer):
    default_name = 'mask_head'

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
            trainable=True,
            name=None
        ):
        self._init(locals())
        super(MaskHead, self).__init__(trainable=trainable, name=name)
        kernel_size = (3, 3)
        fan_out = 256
        init_stddev = MaskHead._get_stddev_equivalent_to_msra_fill(kernel_size, fan_out)
        for conv_id in range(4):
            self._layers[f'mask_conv{str(conv_id)}'] = Conv2D(
                fan_out,
                kernel_size=kernel_size,
                strides=(1, 1),
                padding='same',
                kernel_initializer=tf.random_normal_initializer(stddev=init_stddev),
                use_bias=True,
                bias_initializer='zeros',
                name=f'mask_conv{str(conv_id)}'
            )

        kernel_size = (2, 2)
        fan_out = 256

        init_stddev = MaskHead._get_stddev_equivalent_to_msra_fill(kernel_size, fan_out)
        self._layers['mask_deconv'] = Conv2DTranspose(
            fan_out,
            kernel_size=kernel_size,
            strides=(2, 2),
            padding='valid',
            kernel_initializer=tf.random_normal_initializer(stddev=init_stddev),
            use_bias=True,
            bias_initializer='zeros',
            name='mask_deconv'
        )

        kernel_size = (1, 1)
        fan_out = self.num_classes

        init_stddev = MaskHead._get_stddev_equivalent_to_msra_fill(kernel_size, fan_out)
        self._layers['mask_fcn_logits'] = Conv2D(
            fan_out,
            kernel_size=kernel_size,
            strides=(1, 1),
            padding='valid',
            kernel_initializer=tf.random_normal_initializer(stddev=init_stddev),
            use_bias=True,
            bias_initializer='zeros',
            name='mask_fcn_logits'
        )
    
    @need_build
    def call(self, 
            mask_roi_features, 
            class_indices, 
            training=None):
        indices_dtype = tf.int32
        class_indices = tf.cast(class_indices, dtype=indices_dtype)
        batch_size, num_rois, height, width, filters = mask_roi_features.get_shape().as_list()

        net = tf.reshape(mask_roi_features, [batch_size * num_rois, height, width, filters])
        for conv_id in range(4):
            net = self._layers[f'mask_conv{str(conv_id)}'](net, training=training)
            net = tf.nn.relu(net)
        
        net = self._layers['mask_deconv'](net, training=training)
        net = tf.nn.relu(net)

        mask_outputs = self._layers['mask_fcn_logits'](net, training=training)

        mask_outputs = tf.reshape(
            mask_outputs,
            [-1, num_rois, self.mrcnn_resolution, self.mrcnn_resolution, self.num_classes]
        )

        with tf.name_scope('masks_post_processing'):

            mask_outputs = tf.transpose(a=mask_outputs, perm=[0, 1, 4, 2, 3])

            if batch_size == 1:
                indices = tf.reshape(
                    tf.reshape(
                        tf.range(num_rois, dtype=indices_dtype),
                        [batch_size, num_rois, 1]
                    ) * self.num_classes + tf.expand_dims(class_indices, axis=-1),
                    [batch_size, -1]
                )

                mask_outputs = tf.gather(
                    tf.reshape(mask_outputs,
                               [batch_size, -1, self.mrcnn_resolution, self.mrcnn_resolution]),
                    indices,
                    axis=1
                )

                mask_outputs = tf.squeeze(mask_outputs, axis=1)
                mask_outputs = tf.reshape(
                    mask_outputs,
                    [batch_size, num_rois, self.mrcnn_resolution, self.mrcnn_resolution])

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
    
    def _build(self, input_shape):
        batch_size, num_rois, height, width, filters = input_shape
        resized_input_shape = [batch_size * num_rois if num_rois != None else None, height, width, filters]
        with tf.name_scope(self.name):
            build_layers(
                [self._layers[f'mask_conv{str(conv_id)}'] for conv_id in range(4)] + [self._layers['mask_deconv'], self._layers['mask_fcn_logits']],
                resized_input_shape
            )
        self._output_specs = self._layers['mask_fcn_logits'].output_specs

    def compute_output_specs(self, input_shape):
        pass