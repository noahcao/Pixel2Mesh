from tensorflow.python.training.adam import AdamOptimizer

from p2m.models.gcn import GCN
from p2m.models.losses import gcn_loss


class Trainer(object):
    def __init__(self, config):
        self.config = config
        self.model = GCN(config)
        self.optimizer = AdamOptimizer(learning_rate=config.TRAIN.LEARNING_RATE)
        self.model.add_loss(gcn_loss)

    def train(self):
        self.model.compile(optimizer=self.optimizer)
        self.model.fit()
