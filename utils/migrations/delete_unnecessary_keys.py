from argparse import ArgumentParser

import torch

parser = ArgumentParser()
parser.add_argument("--input", type=str, required=True)
parser.add_argument("--output", type=str, required=True)
args = parser.parse_args()

data = torch.load(args.input)
compressed = dict()
compressed["model"] = data["model"]
torch.save(compressed, args.output)
