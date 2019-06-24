import os
import yaml

import numpy as np
from easydict import EasyDict as edict

config = edict()

config.OUTPUT_DIR = 'output'
config.LOG_DIR = 'log'
config.DATA_DIR = 'data/ShapeNetP2M'
config.WORKERS = 4
config.PRINT_FREQ = 20

config.MODEL = edict()
config.MODEL.HIDDEN_DIM = 256  # gcn hidden layer channel
config.MODEL.FEAT_DIM = 963  # Number of units in feature layer, image feature dim
config.MODEL.COORD_DIM = 3  # Number of units in output layer

config.TRAIN = edict()
config.TRAIN.LEARNING_RATE = 1E-5
config.TRAIN.BEGIN_EPOCH = 0
config.TRAIN.END_EPOCH = 5
config.TRAIN.WEIGHT_DECAY = 5E-6  # Weight decay for L2 loss


def _update_dict(full_key, val, d):
    for vk, vv in val.items():
        if vk not in d:
            raise ValueError("{}.{} does not exist in config".format(full_key, vk))
        if isinstance(vv, list):
            d[vk] = np.array(vv)
        elif isinstance(vv, dict):
            _update_dict(full_key + "." + vk, vv, d[vk])
        else:
            d[vk] = vv


def _update_config(config_file):
    # do scan twice
    # in the first round, MODEL.NAME is located so that we can initialize MODEL.EXTRA
    # in the second round, we update everything

    with open(config_file) as f:
        config_dict = yaml.load(f)
        # do a dfs on `BASED_ON` config files
        if "BASED_ON" in config_dict:
            for base_config in config_dict["BASED_ON"]:
                _update_config(os.path.join(os.path.dirname(config_file), base_config))
            config_dict.pop("BASED_ON")
        _update_dict("", config_dict, config)


def update_config(config_file):
    _update_config(config_file)


def gen_config(config_file):
    cfg = dict(config)
    for k, v in cfg.items():
        if isinstance(v, edict):
            cfg[k] = dict(v)

    with open(config_file, 'w') as f:
        yaml.dump(dict(cfg), f, default_flow_style=False)
