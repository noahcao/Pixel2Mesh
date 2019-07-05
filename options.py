import os
import pprint
from datetime import datetime

import numpy as np
import yaml
from easydict import EasyDict as edict
from tensorboardX import SummaryWriter

from logger import create_logger

options = edict()

options.name = 'p2m'
options.version = None
options.num_workers = 1
options.num_gpus = 1
options.pin_memory = True

options.log_dir = "logs"
options.summary_dir = "summary"
options.checkpoint_dir = "checkpoints"
options.checkpoint = None

options.dataset = edict()
options.dataset.name = "shapenet"
options.dataset.subset_train = "train_small"
options.dataset.subset_eval = "test_small"

options.dataset.shapenet = edict()
options.dataset.shapenet.num_points = 3000

options.model = edict()
options.model.name = "pixel2mesh"
options.model.hidden = 192
options.model.coord_dim = 3

options.loss = edict()
options.loss.weights = edict()
options.loss.weights.normal = 1.
options.loss.weights.edge = 1.
options.loss.weights.laplace = 1.

options.train = edict()
options.train.num_epochs = 200
options.train.batch_size = 4
options.train.summary_steps = 50
options.train.checkpoint_steps = 10000
options.train.test_steps = 10000
options.train.rot_factor = 30
options.train.noise_factor = 0.4
options.train.scale_factor = 0.25
options.train.use_augmentation = True
options.train.use_augmentation_rgb = True
options.train.shuffle = True

options.test = edict()
options.test.dataset = []
options.test.summary_steps = 50
options.test.batch_size = 4
options.test.shuffle = True

options.optim = edict()
options.optim.adam_beta1 = 0.9
options.optim.lr = 2.5e-4
options.optim.wd = 5e-6
options.optim.lr_step = [140, 180]
options.optim.lr_factor = 0.1


def _update_dict(full_key, val, d):
    for vk, vv in val.items():
        if vk not in d:
            raise ValueError("{}.{} does not exist in options".format(full_key, vk))
        if isinstance(vv, list):
            d[vk] = np.array(vv)
        elif isinstance(vv, dict):
            _update_dict(full_key + "." + vk, vv, d[vk])
        else:
            d[vk] = vv


def _update_options(options_file):
    # do scan twice
    # in the first round, MODEL.NAME is located so that we can initialize MODEL.EXTRA
    # in the second round, we update everything

    with open(options_file) as f:
        options_dict = yaml.load(f)
        # do a dfs on `BASED_ON` options files
        if "based_on" in options_dict:
            for base_options in options_dict["based_on"]:
                _update_options(os.path.join(os.path.dirname(options_file), base_options))
            options_dict.pop("based_on")
        _update_dict("", options_dict, options)


def update_options(options_file):
    _update_options(options_file)


def gen_options(options_file):
    cfg = dict(options)
    for k, v in cfg.items():
        if isinstance(v, edict):
            cfg[k] = dict(v)

    with open(options_file, 'w') as f:
        yaml.dump(dict(cfg), f, default_flow_style=False)


def reset_options(options, args, phase='train'):
    if args.batch_size:
        options.train.batch_size = options.test.batch_size = args.batch_size
    if args.version:
        options.version = args.version
    if hasattr(args, "num_epochs") and args.num_epochs:
        options.train.num_epochs = args.num_epochs
    if hasattr(args, "checkpoint") and args.checkpoint:
        options.checkpoint = args.checkpoint

    options.name = args.name

    if options.version is None:
        options.version = datetime.now().strftime('%Y%m%d%H%M%S')
    options.log_dir = os.path.join(options.log_dir, options.name)
    print('=> creating {}'.format(options.log_dir))
    os.makedirs(options.log_dir, exist_ok=True)

    options.checkpoint_dir = os.path.join(options.checkpoint_dir, options.name, options.version)
    print('=> creating {}'.format(options.checkpoint_dir))
    os.makedirs(options.checkpoint_dir, exist_ok=True)

    options.summary_dir = os.path.join(options.summary_dir, options.name, options.version)
    print('=> creating {}'.format(options.summary_dir))
    os.makedirs(options.summary_dir, exist_ok=True)

    logger = create_logger(options, phase=phase)
    logger.info(pprint.pformat(vars(options)))

    print('=> creating summary writer')
    writer = SummaryWriter(options.summary_dir)

    return logger, writer
