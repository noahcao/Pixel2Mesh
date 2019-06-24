import tensorflow as tf
from tensorflow.python.keras import activations, initializers
from tensorflow.python.keras.layers import Layer, Dropout

from p2m.shortcuts import sparse_dropout, dot


class GraphConvolution(Layer):
    """Graph convolution layer."""

    def __init__(self, input_dim, output_dim, support, dropout=0.,
                 sparse_inputs=False, activation="relu",
                 bias=True, featureless=False, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)
        self.dropout = Dropout(rate=dropout)

        self.activations = activations.get(activation)
        self.support = support
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless

        # helper variable for sparse dropout
        self.num_features_nonzero = 3  # placeholders['num_features_nonzero']

        self.w = []
        for i in range(len(self.support)):
            self.w.append(self.add_weight("weights_%d" % i, [input_dim, output_dim], dtype=tf.float32,
                                          initializer=initializers.glorot_uniform))
        self.bias = self.add_weight("bias", [output_dim], dtype=tf.float32) if bias else None

    def call(self, inputs, *args):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1 - self.dropout.rate, self.num_features_nonzero)
        else:
            x = self.dropout(x)

        # convolve
        supports = []
        for i in range(len(self.support)):
            if self.featureless:
                pre_sup = self.w[i]
            else:
                pre_sup = dot(x, self.w[i], sparse=self.sparse_inputs)
            support = dot(self.support[i], pre_sup, sparse=True)
            supports.append(support)
        output = tf.add_n(supports)

        if self.bias:
            output += self.bias
        return self.activations(output)
