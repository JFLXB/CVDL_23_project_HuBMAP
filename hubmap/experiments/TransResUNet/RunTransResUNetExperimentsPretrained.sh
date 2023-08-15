#!/bin/bash

mkdir -p ./logs/final/pretrained

echo "PRETRAINED"
python ./hubmap/experiments/TransResUNet/train.py\
    --name resnet50_pretrained \
    --epochs 20 \
    --backbone resnet50 \
    --pretrained True \
    --model TransResUNet \
    --use-lr-scheduler False \
    --lrs-patience None \
    --use-early-stopping False \
    --es-patience None \
    --loss DiceBCELoss \
    --weights None \