import os

import torch
import torch.nn


class CheckpointSaver(object):
    """Class that handles saving and loading checkpoints during training."""

    def __init__(self, logger, checkpoint_dir=None, checkpoint_file=None):
        self.logger = logger
        if checkpoint_file is not None:
            if not os.path.exists(checkpoint_file):
                raise ValueError("Checkpoint file [%s] does not exist!" % checkpoint_file)
            self.save_dir = os.path.dirname(os.path.abspath(checkpoint_file))
            self.checkpoint_file = os.path.abspath(checkpoint_file)
            return
        if checkpoint_dir is None:
            raise ValueError("Checkpoint directory must be not None in case file is not provided!")
        self.save_dir = os.path.abspath(checkpoint_dir)
        self.checkpoint_file = self.get_latest_checkpoint()

    def load_checkpoint(self):
        if self.checkpoint_file is None:
            self.logger.info("Checkpoint file not found, skipping...")
            return None
        self.logger.info("Loading checkpoint file: %s" % self.checkpoint_file)
        try:
            return torch.load(self.checkpoint_file)
        except UnicodeDecodeError:
            # to be compatible with old encoding methods
            return torch.load(self.checkpoint_file, encoding="bytes")

    def save_checkpoint(self, obj, name):
        self.checkpoint_file = os.path.join(self.save_dir, "%s.pt" % name)
        self.logger.info("Dumping to checkpoint file: %s" % self.checkpoint_file)
        torch.save(obj, self.checkpoint_file)

    def get_latest_checkpoint(self):
        # this will automatically find the checkpoint with latest modified time
        checkpoint_list = []
        for dirpath, dirnames, filenames in os.walk(self.save_dir):
            for filename in filenames:
                if filename.endswith('.pt'):
                    file_path = os.path.abspath(os.path.join(dirpath, filename))
                    modified_time = os.path.getmtime(file_path)
                    checkpoint_list.append((file_path, modified_time))
        checkpoint_list = sorted(checkpoint_list, key=lambda x: x[1])
        return None if not checkpoint_list else checkpoint_list[-1][0]
