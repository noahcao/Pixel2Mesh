import tensorflow as tf

from p2m.layers.base import Layer


class GraphPooling(Layer):
    """Graph Pooling layer."""

    def __init__(self, placeholders, pool_id=1, **kwargs):
        super(GraphPooling, self).__init__(**kwargs)

        self.pool_idx = placeholders['pool_idx'][pool_id - 1]

    def _call(self, inputs):
        X = inputs

        add_feat = (1 / 2.0) * tf.reduce_sum(tf.gather(X, self.pool_idx), 1)
        outputs = tf.concat([X, add_feat], 0)

        return outputs