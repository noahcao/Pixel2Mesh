import tensorflow as tf
from tensorflow.python.keras import activations, initializers
from tensorflow.python.keras.layers import Layer, Dropout

from p2m.layers.graph_convolution import GraphConvolution


class GraphConvResBlock(Layer):
    """
    GCNN Residual Block,
    keeping dimension unchanged
    """

    def __init__(self, dim, support, **kwargs):
        super(GraphConvResBlock, self).__init__(**kwargs)
        self.conv1 = GraphConvolution(dim, dim, support, **kwargs)
        self.conv2 = GraphConvolution(dim, dim, support, **kwargs)

    def call(self, inputs, *args):
        x = self.conv1(inputs)
        x = self.conv2(x)
        return tf.add(x, inputs) * 0.5
