import tensorflow as tf
import numpy as np
# m.update_state([0, 0, 1, 1], [0, 1, 0, 1])
# m([0, 0, 1, 1], [0, 1, 0, 1])
class M(tf.keras.layers.Layer):
    def __init__(self) -> None:
        super().__init__()
        self.metric = tf.keras.metrics.MeanIoU(num_classes=2)
    def call(self, x, y):
        self.add_metric(self.metric(x, y))
        tf.print(self.metric.result())
        return x + y

m = M()

print(m(np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1])))
print(m(np.array([0, 1, 1, 1]), np.array([0, 1, 0, 1])))

# print(m.metric.result().numpy())
print(m.metrics[0].result().numpy())