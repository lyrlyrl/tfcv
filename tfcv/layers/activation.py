import tensorflow as tf

ACTIVATION = {
    'relu': tf.nn.relu
}

def get_activation(name: str):
    assert name in ACTIVATION
    return ACTIVATION[name]