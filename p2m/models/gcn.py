import tensorflow as tf
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine.sequential import Sequential
from tensorflow.python.keras.engine.training import Model
from tensorflow.python.keras.layers import Conv2D

from p2m.layers.graph_conv_resblock import GraphConvResBlock
from p2m.layers.graph_convolution import GraphConvolution
from p2m.layers.graph_pooling import GraphPooling
from p2m.layers.graph_projection import GraphProjection
from p2m.models.losses import mesh_loss, laplace_loss


class GCN(Model):
    def __init__(self, config, **kwargs):
        super(GCN, self).__init__(**kwargs)
        self.build(config)

    def call(self, inputs, *args):
        img_feats = self.call_cnn18(inputs["img_input"])
        x = inputs["features"]
        outputs, outputs_unpool = [], []
        x_conv = None
        for i, (gcn_layer, proj, final_conv) in enumerate(zip(self.gcn_layers, self.projections,
                                                              self.final_conv)):
            x_proj = proj((x, img_feats))
            if i > 0:
                outputs_unpool.append(x_proj)
            assert x_conv is not None
            x_proj_concat = tf.concat([x_proj, x_conv])
            x_conv = gcn_layer(x_proj_concat)
            x = final_conv(x_conv)
            outputs.append(x)
        return {
            "outputs": outputs,
            "outputs_unpool": outputs_unpool
        }

    def call_loss(self, inputs, outputs, outputs_unpool):

        # Weight decay loss
        conv_layers = range(1, 15) + range(17, 31) + range(33, 48)
        for layer_id in conv_layers:
            for var in self.layers[layer_id].vars.values():
                self.loss += config.TRAIN.WEIGHT_DECAY * tf.nn.l2_loss(var)

    def _build(self, config):
        self.build_cnn18()  # update image feature

        self.gcn_layers = []
        self.projections = []
        self.final_conv = []
        for i in range(3):
            self.projections.append(GraphProjection())
            layers = []
            if i > 0:
                layers.append(self.layers.append(GraphPooling(self.pool_idx[i - 1])))
            layers.append(GraphConvolution(input_dim=config.MODEL.FEAT_DIM + (config.MODEL.HIDDEN_DIM if i > 0 else 0),
                                           output_dim=config.MODEL.HIDDEN_DIM, support=self.support[i]))
            for _ in range(6):
                layers.append(GraphConvResBlock(dim=config.MODEL.HIDDEN_DIM, support=self.support[i]))
            if i == 2:
                self.final_conv.append(Sequential([
                    GraphConvolution(input_dim=config.MODEL.HIDDEN_DIM, output_dim=config.MODEL.HIDDEN_DIM // 2,
                                     support=self.support[i]),
                    GraphConvolution(input_dim=config.MODEL.HIDDEN_DIM // 2, output_dim=config.MODEL.COORD_DIM,
                                     support=self.support[i], activation=None)
                ]))
            else:
                self.final_conv.append(GraphConvolution(input_dim=config.MODEL.HIDDEN_DIM,
                                                        output_dim=config.MODEL.COORD_DIM,
                                                        support=self.support[i], activation=None))
            self.gcn_layers.append(Sequential(layers=layers, name="projection_layers_%d" % i))

    def build_cnn18(self):
        self.conv, self.stride_conv = [], []
        filter_size = 16
        for i in range(6):
            self.conv.append(Sequential([
                Conv2D(filter_size, (3, 3), strides=1, activation="relu", kernel_regularizer=regularizers.l2(1E-5)),
                Conv2D(filter_size, (3, 3), strides=1, activation="relu", kernel_regularizer=regularizers.l2(1E-5)),
            ]))
            filter_size *= 2
            self.stride_conv.append(Conv2D(filter_size, (3, 3), strides=2, activation="relu",
                                           kernel_regularizer=regularizers.l2(1E-5)))

    def call_cnn18(self, img_input):
        x = tf.expand_dims(img_input, 0)
        feats = []
        for i in range(len(self.conv)):
            x = self.conv[i](x)
            if i >= 2:
                feats.append(x)
            x = self.stride_conv[i](x)
        return feats
