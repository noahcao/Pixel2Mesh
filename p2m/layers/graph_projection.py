import tensorflow as tf

from p2m.layers.base import Layer


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

    def __init__(self, placeholders, **kwargs):
        super(GraphProjection, self).__init__(**kwargs)

        self.img_feat = placeholders['img_feat']

    '''
    def _call(self, inputs):
        coord = inputs
        X = inputs[:, 0]
        Y = inputs[:, 1]
        Z = inputs[:, 2]

        #h = (-Y)/(-Z)*248 + 224/2.0 - 1
        #w = X/(-Z)*248 + 224/2.0 - 1 [28,14,7,4]
        h = 248.0 * tf.divide(-Y, -Z) + 112.0
        w = 248.0 * tf.divide(X, -Z) + 112.0

        h = tf.minimum(tf.maximum(h, 0), 223)
        w = tf.minimum(tf.maximum(w, 0), 223)
        indeces = tf.stack([h,w], 1)

        idx = tf.cast(indeces/(224.0/56.0), tf.int32)
        out1 = tf.gather_nd(self.img_feat[0], idx)
        idx = tf.cast(indeces/(224.0/28.0), tf.int32)
        out2 = tf.gather_nd(self.img_feat[1], idx)
        idx = tf.cast(indeces/(224.0/14.0), tf.int32)
        out3 = tf.gather_nd(self.img_feat[2], idx)
        idx = tf.cast(indeces/(224.0/7.00), tf.int32)
        out4 = tf.gather_nd(self.img_feat[3], idx)

        outputs = tf.concat([coord,out1,out2,out3,out4], 1)
        return outputs
    '''

    def _call(self, inputs):
        coord = inputs
        X = inputs[:, 0]
        Y = inputs[:, 1]
        Z = inputs[:, 2]

        h = 250 * tf.divide(-Y, -Z) + 112
        w = 250 * tf.divide(X, -Z) + 112

        h = tf.minimum(tf.maximum(h, 0), 223)
        w = tf.minimum(tf.maximum(w, 0), 223)

        x = h / (224.0 / 56)
        y = w / (224.0 / 56)
        out1 = project(self.img_feat[0], x, y, 64)

        x = h / (224.0 / 28)
        y = w / (224.0 / 28)
        out2 = project(self.img_feat[1], x, y, 128)

        x = h / (224.0 / 14)
        y = w / (224.0 / 14)
        out3 = project(self.img_feat[2], x, y, 256)

        x = h / (224.0 / 7)
        y = w / (224.0 / 7)
        out4 = project(self.img_feat[3], x, y, 512)
        outputs = tf.concat([coord, out1, out2, out3, out4], 1)
        return outputs