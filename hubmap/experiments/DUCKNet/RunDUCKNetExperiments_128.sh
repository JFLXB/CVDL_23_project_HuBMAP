#!/bin/bash

mkdir -p ./logs/final/ducknet

python ./hubmap/experiments/DUCKNet/train.py\
    --name DUCKNet \
    --epochs 25 \
    --img-size 128 \
    --model DUCKNet \
    --use-lr-scheduler \
    --lrs-patience 10 \
    --use-early-stopping \
    --loss DiceLoss \