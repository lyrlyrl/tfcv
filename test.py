import tensorflow as tf

class Holder(tf.Module):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.w = tf.Variable(initial_value=1, trainable=True)
        self.b = tf.Variable(initial_value=1, trainable=False)
    def __call__(self, inputs):
        self.w.assign_add(inputs)
        self.b.assign_add(inputs)
        return inputs

h = Holder()
print(h.w, h.b)
h(1)
print(h.w, h.b)