import pickle

import numpy as np
import tensorflow as tf
from tensorflow.python.training.adam import AdamOptimizer

from p2m.dataset import create_dataset, shapenet_p2m_process
from p2m.models.gcn import GCN
from p2m.models.losses import gcn_loss


class Trainer(object):
    def __init__(self, config):
        self.config = config

        self.initial_params = dict()
        with open('data/ellipsoid/info_ellipsoid.dat', 'rb') as info_ellipsoid:
            pkl = pickle.load(info_ellipsoid, encoding="latin1")

        num_blocks = 3
        num_supports = 2
        self.initial_params["features"] = tf.convert_to_tensor(pkl[0])  # coord
        for i in range(1, 4):
            self.initial_params["edges_%d" % (i - 1)] = tf.convert_to_tensor(pkl[i][1][0])
        for i, t in enumerate(pkl[5]):
            self.initial_params["faces_%d" % i] = tf.convert_to_tensor(t)
        for i, t in enumerate(pkl[7][:num_blocks]):
            self.initial_params["lape_idx_%d" % i] = tf.convert_to_tensor(t)
        pool_idx = [tf.convert_to_tensor(t) for t in pkl[4][:num_blocks - 1]]
        support = [[tf.SparseTensor(*pkl[i + 1][j]) for j in range(num_supports)] for i in range(3)]

        self.model = GCN(config, pool_idx, support)
        self.optimizer = AdamOptimizer(learning_rate=config.TRAIN.LEARNING_RATE)

        self.dataset = create_dataset("data/train_list.txt") \
            .map(lambda filename: tf.py_func(shapenet_p2m_process, [filename], [tf.float32, tf.float32, tf.string])) \
            .map(self.update_data_with_initial_params)
        self.model.compile(optimizer=self.optimizer)

    def update_data_with_initial_params(self, img, labels, data_id):
        data = self.initial_params.copy()
        data["img_input"] = img
        data["labels"] = labels
        data["data_id"] = data_id
        return data

    def train(self):
        for epoch in range(3):
            print('Start of epoch %d' % epoch)
            for step, batch in enumerate(self.dataset):
                with tf.GradientTape() as tape:
                    outputs = self.model(batch)
                    loss_value = gcn_loss(batch, outputs)

                grads = tape.gradient(loss_value, self.model.trainable_weights)
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

                # Log every 200 batches.
                if step % 200 == 0:
                    print('Training loss (for one batch) at step %s: %s' % (step, float(loss_value)))
                    print('Seen so far: %s samples' % ((step + 1) * 64))
