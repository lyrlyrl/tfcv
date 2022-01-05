import tensorflow as tf

from tfcv.layers.base import Layer

from tfcv.layers.conv2d import Conv2D
from tfcv.layers.normalization import BatchNormalization
from tfcv.layers.linear import Linear

class RPNHead(Layer):

    default_name = 'rpn_head'

    def __init__(
        self, 
        num_anchors: int,
        num_filters: int,
        trainable=True, name=None):
        self._init(locals())
        super().__init__(trainable=trainable, name=name)
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
        net = self._layers['rpn_conv'](inputs, training)
        net = tf.nn.relu(net)
        scores = self._layers['rpn_score'](net, training)
        bboxes = self._layers['rpn_box'](net, training)

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
            bias_initializer=tf.keras.initializers.Zeros(),
            trainable=trainable, name='class_predict')
        self._layers['box_predict'] = Linear(
            num_classes * 4, 
            kernel_initializer=tf.random_normal_initializer(stddev=0.001),
            bias_initializer=tf.keras.initializers.Zeros(),
            trainable=trainable, name='box_predict')

    def _build(self, input_shape):
        batch_size, num_rois, height, width, filters = input_shape
        self._layers['fc6'].build([batch_size, num_rois, height * width * filters])
        self._layers['fc7'].build(self._layers['fc6'].output_specs)
        self._layers['class_predict'].build(self._layers['fc7'].output_specs)
        self._layers['box_predict'].build(self._layers['fc7'].output_specs)
        self._output_specs = (self._layers['class_predict'].output_specs, 
            self._layers['box_predict'].output_specs)

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