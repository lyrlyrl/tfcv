import tensorflow as tf

from tfcv.layers.linear import Linear

if __name__ == '__main__':
    layer = Linear(2, 4)

    inputs = tf.ones([2,2,4])
    layer.build(inputs)
    print(layer.inference_forward(inputs))
    print(layer.output_specs)
    
    print(layer.output_specs)
    print(layer.get_layer('test'))