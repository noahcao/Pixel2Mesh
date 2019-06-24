import tensorflow as tf
from tensorflow.python.keras.engine import Layer


class GraphPooling(Layer):

    def __init__(self, pool_idx, **kwargs):
        super(GraphPooling, self).__init__(**kwargs)
        self.pool_idx = pool_idx

    def call(self, inputs, *args):
        add_feat = (1 / 2.0) * tf.reduce_sum(tf.gather(inputs, self.pool_idx), 1)
        outputs = tf.concat([inputs, add_feat], 0)

        return outputs
