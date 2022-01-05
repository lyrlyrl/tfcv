from tfcv.models.experimental.resnet import ResNet
from tfcv.models.resnet import ResNet as OResNet
import tensorflow as tf
import numpy as np
if __name__ == '__main__':
    model = ResNet(50, include_top=False)
    model.build([4, 224, 224, 3])
    model.load_weights('/tmp/.mlco/pretrained_weights/resnet50_imagenet.npz')
    inputs = tf.ones([4, 224, 224, 3])
    output1 = model(inputs, training=False)
    output1 = tf.nest.map_structure(lambda x:x.numpy(), output1)
    
    model_o = OResNet(50, [224, 224, 3], include_top=False)
    output2 = model_o(inputs, training=False)
    output2 = tf.nest.map_structure(lambda x: x.numpy(), output2)

    for k, v in output1.items():
        assert (v == output2[k]).all()
        print(v[0, 0, 0, 0:10], output2[k][0, 0, 0, 0:10])
        # break
