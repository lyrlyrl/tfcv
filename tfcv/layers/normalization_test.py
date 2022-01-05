from tfcv.layers.normalization import BatchNormalization
import tensorflow as tf

if __name__ == '__main__':
    bn = BatchNormalization()
    bnl = tf.keras.layers.BatchNormalization(axis=-1)

    bn.build([None, 14, 14, 3])
    for v in bn.variables:
        print(v.name, v.shape)
    
    bnl.build([None, 14, 14, 3])
    for v in bnl.variables:
        print(v.name, v.shape)