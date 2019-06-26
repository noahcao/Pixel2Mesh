#!/usr/bin/env bash

set -x

PARTITION=HA_3D
NAME=$1

srun -p ${PARTITION} \
    --job-name=CMR \
    --gres=gpu:8 \
    --ntasks=1 \
    --kill-on-bad-exit=1 \
    python entrypoint_train.py --name ${NAME} --batch-size 16 &
