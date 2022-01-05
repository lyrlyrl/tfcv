from tfcv.models.experimental.resnet import ResNet
from tfcv.models.experimental.fpn import FPN
import tensorflow as tf

if __name__ == '__main__':
    model = ResNet(50, include_top=False)
    model.build([None, 224, 224, 3])
    model_fpn = FPN()
    model_fpn.build(model.output_specs)
    inputs = tf.ones([4, 224, 224, 3])
    output1 = model(inputs, training=False)
    output2 = model_fpn(output1, training=False)
    print(model_fpn.output_specs)
    for k, v in output2.items():
        print(k, v.shape)