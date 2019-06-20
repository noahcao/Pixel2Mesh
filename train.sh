#!/usr/bin/env bash

set -x

PARTITION=HA_3D

srun -p ${PARTITION} \
    --job-name=Mesh \
    --gres=gpu:1 \
    --ntasks=1 \
    --kill-on-bad-exit=1 \
    python train.py &
