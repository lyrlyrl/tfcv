import tensorflow as tf
from tfcv import G

print(G.train_metrics)

class MyDense(tf.keras.layers.Layer):
    # Adding **kwargs to support base Keras layer arguments
    def __init__(self, in_features, out_features, **kwargs):
        super().__init__(**kwargs)

        # This will soon move to the build step; see below
        self.w = tf.Variable(
            tf.random.normal([in_features, out_features]), name='w')
        self.b = tf.Variable(tf.ones([out_features]), name='b')

        G.train_metrics['mean'] = tf.keras.metrics.Mean('mean')
    def call(self, x):
        if G.is_training:
            y = tf.matmul(x, self.w) + self.b
        else:
            y = tf.matmul(x, self.w) - self.b
        G.train_metrics['mean'].update_state(tf.reduce_sum(y))
        return tf.nn.relu(y)

simple_layer = MyDense(name="simple", in_features=3, out_features=3)

res1 = simple_layer([[2.0, 2.0, 2.0]])

print(res1)

G.is_training = False

res2 = simple_layer([[2.0, 2.0, 2.0]])

print(res2)
# print(simple_layer.acc.result())
print(G.train_metrics['mean'].result())
print(G.dsad)