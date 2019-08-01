import argparse
import sys

from functions.predictor import Predictor
from options import update_options, options, reset_options


def parse_args():
    parser = argparse.ArgumentParser(description='Pixel2Mesh Prediction Entrypoint')
    parser.add_argument('--options', help='experiment options file name', required=False, type=str)

    args, rest = parser.parse_known_args()
    if args.options is None:
        print("Running without options file...", file=sys.stderr)
    else:
        update_options(args.options)

    parser.add_argument('--batch-size', help='batch size', type=int)
    parser.add_argument('--checkpoint', help='trained model file', type=str, required=True)
    parser.add_argument('--name', required=True, type=str)
    parser.add_argument('--folder', required=True, type=str)

    options.dataset.name += '_demo'

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    logger, writer = reset_options(options, args, phase='predict')

    predictor = Predictor(options, logger, writer)
    predictor.predict()


if __name__ == "__main__":
    main()
