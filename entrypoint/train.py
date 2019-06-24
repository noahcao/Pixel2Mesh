import argparse
import sys

import tensorflow as tf

from p2m.config import update_config, config
from p2m.trainer.base import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description='Pixel2Mesh Train Entrypoint')
    parser.add_argument('--cfg', help='experiment configure file name', required=False, type=str)

    args, rest = parser.parse_known_args()
    if args.cfg is None:
        print("Running without configuration file...", file=sys.stderr)
    else:
        update_config(args.cfg)

    # training
    parser.add_argument('--frequent', help='frequency of logging', default=config.PRINT_FREQ, type=int)
    parser.add_argument('--batch-size', help='batch size', type=int)
    parser.add_argument("--checkpoint", help="checkpoint file", type=str)

    args = parser.parse_args()

    return args


def reset_config(config, args):
    if args.batch_size:
        config.TRAIN.BATCH_SIZE = config.TEST.BATCH_SIZE = args.batch_size
    if args.frequent:
        config.PRINT_FREQ = args.frequent


def main():
    args = parse_args()
    reset_config(config, args)
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    tf.enable_eager_execution()
    main()