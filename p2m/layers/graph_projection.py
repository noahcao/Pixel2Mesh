import tensorflow as tf
from tensorflow.python.keras.engine import Layer


def project(img_feat, x, y, dim):
    x1 = tf.floor(x)
    x2 = tf.ceil(x)
    y1 = tf.floor(y)
    y2 = tf.ceil(y)
    Q11 = tf.gather_nd(img_feat, tf.stack([tf.cast(x1, tf.int32), tf.cast(y1, tf.int32)], 1))
    Q12 = tf.gather_nd(img_feat, tf.stack([tf.cast(x1, tf.int32), tf.cast(y2, tf.int32)], 1))
    Q21 = tf.gather_nd(img_feat, tf.stack([tf.cast(x2, tf.int32), tf.cast(y1, tf.int32)], 1))
    Q22 = tf.gather_nd(img_feat, tf.stack([tf.cast(x2, tf.int32), tf.cast(y2, tf.int32)], 1))

    weights = tf.multiply(tf.subtract(x2, x), tf.subtract(y2, y))
    Q11 = tf.multiply(tf.tile(tf.reshape(weights, [-1, 1]), [1, dim]), Q11)

    weights = tf.multiply(tf.subtract(x, x1), tf.subtract(y2, y))
    Q21 = tf.multiply(tf.tile(tf.reshape(weights, [-1, 1]), [1, dim]), Q21)

    weights = tf.multiply(tf.subtract(x2, x), tf.subtract(y, y1))
    Q12 = tf.multiply(tf.tile(tf.reshape(weights, [-1, 1]), [1, dim]), Q12)

    weights = tf.multiply(tf.subtract(x, x1), tf.subtract(y, y1))
    Q22 = tf.multiply(tf.tile(tf.reshape(weights, [-1, 1]), [1, dim]), Q22)

    outputs = tf.add_n([Q11, Q21, Q12, Q22])
    return outputs


class GraphProjection(Layer):
    """Graph Pooling layer."""

    def call(self, inputs, **kwargs):
        coord, img_feat = inputs
        X = coord[:, 0]
        Y = coord[:, 1]
        Z = coord[:, 2]

        h = 250 * tf.divide(-Y, -Z) + 112
        w = 250 * tf.divide(X, -Z) + 112

        h = tf.minimum(tf.maximum(h, 0), 223)
        w = tf.minimum(tf.maximum(w, 0), 223)

        x = h / (224.0 / 56)
        y = w / (224.0 / 56)
        out1 = project(img_feat[0], x, y, 64)

        x = h / (224.0 / 28)
        y = w / (224.0 / 28)
        out2 = project(img_feat[1], x, y, 128)

        x = h / (224.0 / 14)
        y = w / (224.0 / 14)
        out3 = project(img_feat[2], x, y, 256)

        x = h / (224.0 / 7)
        y = w / (224.0 / 7)
        out4 = project(img_feat[3], x, y, 512)
        outputs = tf.concat([coord, out1, out2, out3, out4], 1)
        return outputs
