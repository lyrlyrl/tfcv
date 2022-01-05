import tensorflow as tf

from tfcv.layers.conv2d import Conv2D

if __name__ == '__main__':
    layer = Conv2D(
        filters=64, 
        kernel_size=7, 
        strides=2, 
        padding='same', 
        kernel_initializer='glorot_uniform',
        use_bias=True,
        bias_initializer='glorot_uniform')
    print(layer.kernel_size)

    input_shape = (4, 28, 28, 3)
    x = tf.random.normal(input_shape)

    layer.build(input_shape)

    print(layer.train_forward(x).shape)
    print(layer.kernel.shape, layer.bias.shape)

    tfl = tf.keras.layers.Conv2D(
        filters=64, 
        kernel_size=7, 
        strides=2, 
        padding='same',
        use_bias=True,)
    print(tfl(x).shape)
    for v in tfl.variables:
        print(v.name, v.shape)
