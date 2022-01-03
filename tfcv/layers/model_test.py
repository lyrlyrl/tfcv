from tfcv.layers.base import Layer
from tfcv.layers.linear import Linear
from tfcv.layers.normalization import BatchNormalization
import tensorflow as tf
import numpy as np
class Model(Layer):
    default_name = 'test'
    def __init__(self, name=None):
        super(Model, self).__init__(name=self.default_name)
        self.layer1 = Linear(2, name='linear1')
        self.layer2 = Linear(3, name='linear2')
        self.layer3=BatchNormalization()
        

    def _build(self, input_shape):
        if isinstance(input_shape, np.ndarray):
            input_shape = input_shape.to_list()
        else:
            input_shape = list(input_shape)
        with tf.name_scope(self.name):
            self.layer1.build(input_shape)
            self.layer2.build(self.layer1.output_specs)
            self.layer3.build(self.layer2.output_specs)
            self._output_specs=self.layer3.output_specs
    
    def call(self, inputs, training=None):
        if training:
            return self.layer2.train_forward(
                self.layer1.train_forward(inputs)
            )
        else:
            return self.layer2.inference_forward(
                self.layer1.inference_forward(inputs)
            )

if __name__ == '__main__':
    model = Model()
    print(model.variables)
    model.build([4,4,2])
    print(model.variables)
