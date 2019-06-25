import tensorflow as tf
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine.sequential import Sequential
from tensorflow.python.keras.engine.training import Model
from tensorflow.python.keras.layers import Conv2D

from p2m.layers.graph_conv_resblock import GraphConvResBlock
from p2m.layers.graph_convolution import GraphConvolution
from p2m.layers.graph_pooling import GraphUnpooling
from p2m.layers.graph_projection import GraphProjection


class GCN(Model):
    def __init__(self, config, pool_idx, support, **kwargs):
        super(GCN, self).__init__(**kwargs)

        # build cnn18
        self.conv, self.stride_conv = [], []
        filter_size = 16
        for i in range(6):
            self.conv.append(Sequential([
                Conv2D(filter_size, (3, 3), strides=1, padding="same",
                       activation="relu", kernel_regularizer=regularizers.l2(1E-5)),
                Conv2D(filter_size, (3, 3), strides=1, padding="same",
                       activation="relu", kernel_regularizer=regularizers.l2(1E-5)),
            ]))
            filter_size *= 2
            self.stride_conv.append(Conv2D(filter_size, (3, 3) if i < 3 else (5, 5),
                                           strides=2, padding="same", activation="relu",
                                           kernel_regularizer=regularizers.l2(1E-5)))

        # build gcn layers
        self.gcn_layers = []
        self.projections = []
        self.unpooling_layers = []
        self.final_conv = []
        for i in range(3):
            self.projections.append(GraphProjection())
            self.unpooling_layers.append(GraphUnpooling(pool_idx[i - 1]) if i > 0 else None)
            layers = [GraphConvolution(input_dim=config.MODEL.FEAT_DIM + (config.MODEL.HIDDEN_DIM if i > 0 else 0),
                                       output_dim=config.MODEL.HIDDEN_DIM, support=support[i])]
            for _ in range(6):
                layers.append(GraphConvResBlock(dim=config.MODEL.HIDDEN_DIM, support=support[i]))
            if i == 2:
                self.final_conv.append(Sequential([
                    GraphConvolution(input_dim=config.MODEL.HIDDEN_DIM, output_dim=config.MODEL.HIDDEN_DIM // 2,
                                     support=support[i]),
                    GraphConvolution(input_dim=config.MODEL.HIDDEN_DIM // 2, output_dim=config.MODEL.COORD_DIM,
                                     support=support[i], activation=None)
                ]))
            else:
                self.final_conv.append(GraphConvolution(input_dim=config.MODEL.HIDDEN_DIM,
                                                        output_dim=config.MODEL.COORD_DIM,
                                                        support=support[i], activation=None))
            self.gcn_layers.append(Sequential(layers=layers, name="projection_layers_%d" % i))

    def call(self, inputs, *args):
        img_feats = self.call_cnn18(inputs["img_input"])
        x = inputs["features"]
        outputs, outputs_unpool = [], []
        x_conv = None
        for i, (gcn_layer, proj, unpooling, final_conv) in enumerate(zip(self.gcn_layers, self.projections,
                                                              self.unpooling_layers, self.final_conv)):
            x_proj = proj((x, img_feats))
            if i > 0:
                # append to output unpool
                outputs_unpool.append(unpooling(x))
                assert x_conv is not None
                x_proj = tf.concat([x_proj, x_conv], axis=1)
                x_proj = unpooling(x_proj)
            x_conv = gcn_layer(x_proj)
            x = final_conv(x_conv)
            # append to output
            outputs.append(x)
        return {
            "outputs": outputs,
            "outputs_unpool": outputs_unpool
        }

    def call_cnn18(self, img_input):
        x = tf.expand_dims(img_input, 0)
        feats = []
        for i in range(len(self.conv)):
            x = self.conv[i](x)
            if i >= 2:
                feats.append(tf.squeeze(x))
            x = self.stride_conv[i](x)
        return feats
