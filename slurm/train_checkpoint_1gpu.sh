#!/usr/bin/env bash

set -x

if [[ $# -lt 4 ]] ; then
    echo 'too few arguments supplied'
    exit 1
fi

PARTITION=$1
NAME=$2
OPTIONS=$3
CHECKPOINT=$4

srun -p ${PARTITION} \
    --job-name=Mesh \
    --gres=gpu:1 \
    --ntasks=1 \
    --kill-on-bad-exit=1 \
    python entrypoint_train.py --name ${NAME} --options ${OPTIONS} --checkpoint ${CHECKPOINT} &
