import tensorflow as tf

from tfcv.layers.conv2d import Conv2D

if __name__ == '__main__':
    with tf.name_scope('train'):
        layer = Conv2D(
            filters=2, 
            kernel_size=3, 
            strides=1, 
            padding='same', 
            kernel_initializer='glorot_uniform',
            use_bias=True,
            bias_initializer='glorot_uniform')
        print(layer.kernel_size)

        input_shape = (4, 28, 28, 3)
        x = tf.random.normal(input_shape)

        layer.build(input_shape)

        print(layer.train_forward(x).shape)
        print(layer.kernel, layer.bias)
