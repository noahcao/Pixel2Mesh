import argparse
import sys

from functions.evaluator import Evaluator
from options import update_options, options, reset_options


def parse_args():
    parser = argparse.ArgumentParser(description='GraphCMR Training Entrypoint')
    parser.add_argument('--options', help='experiment options file name', required=False, type=str)

    args, rest = parser.parse_known_args()
    if args.options is None:
        print("Running without options file...", file=sys.stderr)
    else:
        update_options(args.options)

    parser.add_argument('--batch-size', help='batch size', type=int)
    parser.add_argument('--checkpoint', help='trained model file', type=str, required=True)
    parser.add_argument('--dataset', default='lsp', type=str)
    parser.add_argument('--name', required=True, type=str)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    logger = reset_options(options, args, phase='eval')

    # Load model
    if args.dataset == "all":
        datasets = ["h36m-p1", "h36m-p2", "up-3d", "lsp"]
    else:
        datasets = [args.dataset]
    for dataset in datasets:
        evaluator = Evaluator(options, logger, dataset=dataset)
        evaluator.evaluate()


if __name__ == "__main__":
    main()
