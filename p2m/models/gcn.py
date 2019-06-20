import tensorflow as tf
import tflearn

from p2m.layers.graph_convolution import GraphConvolution
from p2m.layers.graph_pooling import GraphPooling
from p2m.layers.graph_projection import GraphProjection
from p2m.losses import mesh_loss, laplace_loss
from p2m.models import Model


class GCN(Model):
    def __init__(self, config, placeholders, **kwargs):
        super(GCN, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=config.TRAIN.LEARNING_RATE)

        self.build(config)

    def _loss(self, config):
        '''
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
        '''
        self.loss += mesh_loss(self.output1, self.placeholders, 1)
        self.loss += mesh_loss(self.output2, self.placeholders, 2)
        self.loss += mesh_loss(self.output3, self.placeholders, 3)
        self.loss += .1 * laplace_loss(self.inputs, self.output1, self.placeholders, 1)
        self.loss += laplace_loss(self.output1_2, self.output2, self.placeholders, 2)
        self.loss += laplace_loss(self.output2_2, self.output3, self.placeholders, 3)

        # Weight decay loss
        conv_layers = range(1, 15) + range(17, 31) + range(33, 48)
        for layer_id in conv_layers:
            for var in self.layers[layer_id].vars.values():
                self.loss += config.TRAIN.WEIGHT_DECAY * tf.nn.l2_loss(var)

    def _build(self, config):
        self.build_cnn18()  # update image feature
        # first project block
        self.layers.append(GraphProjection(placeholders=self.placeholders))
        self.layers.append(GraphConvolution(input_dim=config.MODEL.FEAT_DIM,
                                            output_dim=config.MODEL.HIDDEN_DIM,
                                            gcn_block_id=1,
                                            placeholders=self.placeholders, logging=self.logging))
        for _ in range(12):
            self.layers.append(GraphConvolution(input_dim=config.MODEL.HIDDEN_DIM,
                                                output_dim=config.MODEL.HIDDEN_DIM,
                                                gcn_block_id=1,
                                                placeholders=self.placeholders, logging=self.logging))
        self.layers.append(GraphConvolution(input_dim=config.MODEL.HIDDEN_DIM,
                                            output_dim=config.MODEL.COORD_DIM,
                                            act=lambda x: x,
                                            gcn_block_id=1,
                                            placeholders=self.placeholders, logging=self.logging))
        # second project block
        self.layers.append(GraphProjection(placeholders=self.placeholders))
        self.layers.append(GraphPooling(placeholders=self.placeholders, pool_id=1))  # unpooling
        self.layers.append(GraphConvolution(input_dim=config.MODEL.FEAT_DIM + config.MODEL.HIDDEN_DIM,
                                            output_dim=config.MODEL.HIDDEN_DIM,
                                            gcn_block_id=2,
                                            placeholders=self.placeholders, logging=self.logging))
        for _ in range(12):
            self.layers.append(GraphConvolution(input_dim=config.MODEL.HIDDEN_DIM,
                                                output_dim=config.MODEL.HIDDEN_DIM,
                                                gcn_block_id=2,
                                                placeholders=self.placeholders, logging=self.logging))
        self.layers.append(GraphConvolution(input_dim=config.MODEL.HIDDEN_DIM,
                                            output_dim=config.MODEL.COORD_DIM,
                                            act=lambda x: x,
                                            gcn_block_id=2,
                                            placeholders=self.placeholders, logging=self.logging))
        # third project block
        self.layers.append(GraphProjection(placeholders=self.placeholders))
        self.layers.append(GraphPooling(placeholders=self.placeholders, pool_id=2))  # unpooling
        self.layers.append(GraphConvolution(input_dim=config.MODEL.FEAT_DIM + config.MODEL.HIDDEN_DIM,
                                            output_dim=config.MODEL.HIDDEN_DIM,
                                            gcn_block_id=3,
                                            placeholders=self.placeholders, logging=self.logging))
        for _ in range(12):
            self.layers.append(GraphConvolution(input_dim=config.MODEL.HIDDEN_DIM,
                                                output_dim=config.MODEL.HIDDEN_DIM,
                                                gcn_block_id=3,
                                                placeholders=self.placeholders, logging=self.logging))
        self.layers.append(GraphConvolution(input_dim=config.MODEL.HIDDEN_DIM,
                                            output_dim=int(config.MODEL.HIDDEN_DIM / 2),
                                            gcn_block_id=3,
                                            placeholders=self.placeholders, logging=self.logging))
        self.layers.append(GraphConvolution(input_dim=int(config.MODEL.HIDDEN_DIM / 2),
                                            output_dim=config.MODEL.COORD_DIM,
                                            act=lambda x: x,
                                            gcn_block_id=3,
                                            placeholders=self.placeholders, logging=self.logging))

    def build_cnn18(self):
        x = self.placeholders['img_inp']
        x = tf.expand_dims(x, 0)
        # 224 224
        x = tflearn.layers.conv.conv_2d(x, 16, (3, 3), strides=1, activation='relu', weight_decay=1e-5,
                                        regularizer='L2')
        x = tflearn.layers.conv.conv_2d(x, 16, (3, 3), strides=1, activation='relu', weight_decay=1e-5,
                                        regularizer='L2')
        x0 = x
        x = tflearn.layers.conv.conv_2d(x, 32, (3, 3), strides=2, activation='relu', weight_decay=1e-5,
                                        regularizer='L2')
        # 112 112
        x = tflearn.layers.conv.conv_2d(x, 32, (3, 3), strides=1, activation='relu', weight_decay=1e-5,
                                        regularizer='L2')
        x = tflearn.layers.conv.conv_2d(x, 32, (3, 3), strides=1, activation='relu', weight_decay=1e-5,
                                        regularizer='L2')
        x1 = x
        x = tflearn.layers.conv.conv_2d(x, 64, (3, 3), strides=2, activation='relu', weight_decay=1e-5,
                                        regularizer='L2')
        # 56 56
        x = tflearn.layers.conv.conv_2d(x, 64, (3, 3), strides=1, activation='relu', weight_decay=1e-5,
                                        regularizer='L2')
        x = tflearn.layers.conv.conv_2d(x, 64, (3, 3), strides=1, activation='relu', weight_decay=1e-5,
                                        regularizer='L2')
        x2 = x
        x = tflearn.layers.conv.conv_2d(x, 128, (3, 3), strides=2, activation='relu', weight_decay=1e-5,
                                        regularizer='L2')
        # 28 28
        x = tflearn.layers.conv.conv_2d(x, 128, (3, 3), strides=1, activation='relu', weight_decay=1e-5,
                                        regularizer='L2')
        x = tflearn.layers.conv.conv_2d(x, 128, (3, 3), strides=1, activation='relu', weight_decay=1e-5,
                                        regularizer='L2')
        x3 = x
        x = tflearn.layers.conv.conv_2d(x, 256, (5, 5), strides=2, activation='relu', weight_decay=1e-5,
                                        regularizer='L2')
        # 14 14
        x = tflearn.layers.conv.conv_2d(x, 256, (3, 3), strides=1, activation='relu', weight_decay=1e-5,
                                        regularizer='L2')
        x = tflearn.layers.conv.conv_2d(x, 256, (3, 3), strides=1, activation='relu', weight_decay=1e-5,
                                        regularizer='L2')
        x4 = x
        x = tflearn.layers.conv.conv_2d(x, 512, (5, 5), strides=2, activation='relu', weight_decay=1e-5,
                                        regularizer='L2')
        # 7 7
        x = tflearn.layers.conv.conv_2d(x, 512, (3, 3), strides=1, activation='relu', weight_decay=1e-5,
                                        regularizer='L2')
        x = tflearn.layers.conv.conv_2d(x, 512, (3, 3), strides=1, activation='relu', weight_decay=1e-5,
                                        regularizer='L2')
        x = tflearn.layers.conv.conv_2d(x, 512, (3, 3), strides=1, activation='relu', weight_decay=1e-5,
                                        regularizer='L2')
        x5 = x
        # updata image feature
        self.placeholders.update({'img_feat': [tf.squeeze(x2), tf.squeeze(x3), tf.squeeze(x4), tf.squeeze(x5)]})
        self.loss += tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)) * 0.3