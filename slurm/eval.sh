#!/usr/bin/env bash

set -x

PARTITION=HA_3D
NAME=$1
CHECKPOINT=$2

srun -p ${PARTITION} \
    --job-name=CMR \
    --gres=gpu:8 \
    --ntasks=1 \
    --kill-on-bad-exit=1 \
    python entrypoint_eval.py --name ${NAME} --checkpoint ${CHECKPOINT} --dataset lsp &
