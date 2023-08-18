#!/bin/bash

# mkdir -p ./logs/final/pretrained

# echo "WARMUP x PRETRAINED x 256"
# python ./hubmap/experiments/TransResUNet/train.py\
#     --name resnet50_pretrained_warmup_256 \
#     --epochs 20 \
#     --backbone resnet50 \
#     --pretrained \
#     --model TransResUNet \
#     --loss DiceBCELoss | tee ./logs/final/pretrained/resnet50_pretrained_warmup_256.log

# python ./hubmap/experiments/TransResUNet/train.py\
#     --name resnet101_pretrained_warmup_256 \
#     --epochs 20 \
#     --backbone resnet101 \
#     --pretrained \
#     --model TransResUNet \
#     --loss DiceBCELoss | tee ./logs/final/pretrained/resnet101_pretrained_warmup_256.log

# python ./hubmap/experiments/TransResUNet/train.py\
#     --name resnet152_pretrained_warmup_256 \
#     --epochs 20 \
#     --backbone resnet152 \
#     --pretrained \
#     --model TransResUNet \
#     --loss DiceBCELoss | tee ./logs/final/pretrained/resnet152_pretrained_warmup_256.log

# python ./hubmap/experiments/TransResUNet/train.py\
#     --name resnext50_32x4d_pretrained_warmup_256 \
#     --epochs 20 \
#     --backbone resnext50_32x4d \
#     --pretrained \
#     --model TransResUNet \
#     --loss DiceBCELoss | tee ./logs/final/pretrained/resnext50_32x4d_pretrained_warmup_256.log

# python ./hubmap/experiments/TransResUNet/train.py\
#     --name resnext101_32x8d_pretrained_warmup_256 \
#     --epochs 20 \
#     --backbone resnext101_32x8d \
#     --pretrained \
#     --model TransResUNet \
#     --loss DiceBCELoss | tee ./logs/final/pretrained/resnext101_32x8d_pretrained_warmup_256.log

# python ./hubmap/experiments/TransResUNet/train.py\
#     --name wide_resnet50_2_pretrained_warmup_256 \
#     --epochs 20 \
#     --backbone wide_resnet50_2 \
#     --pretrained \
#     --model TransResUNet \
#     --loss DiceBCELoss | tee ./logs/final/pretrained/wide_resnet50_2_pretrained_warmup_256.log

# python ./hubmap/experiments/TransResUNet/train.py\
#     --name wide_resnet101_2_pretrained_warmup_256 \
#     --epochs 20 \
#     --backbone wide_resnet101_2 \
#     --pretrained \
#     --model TransResUNet \
#     --loss DiceBCELoss | tee ./logs/final/pretrained/wide_resnet101_2_pretrained_warmup_256.log

# mkdir -p ./logs/final/scratch
echo "WARMUP x SCRATCH x 256"
# python ./hubmap/experiments/TransResUNet/train.py\
#     --name resnet50_scratch_warmup_256 \
#     --epochs 20 \
#     --backbone resnet50 \
#     --model TransResUNet \
#     --loss DiceBCELoss | tee ./logs/final/scratch/resnet50_scratch_warmup_256.log

python ./hubmap/experiments/TransResUNet/train.py\
    --name resnet101_scratch_warmup_256 \
    --epochs 20 \
    --backbone resnet101 \
    --model TransResUNet \
    --loss DiceBCELoss | tee ./logs/final/scratch/resnet101_scratch_warmup_256.log

# python ./hubmap/experiments/TransResUNet/train.py\
#     --name resnet152_scratch_warmup_256 \
#     --epochs 20 \
#     --backbone resnet152 \
#     --model TransResUNet \
#     --loss DiceBCELoss | tee ./logs/final/scratch/resnet152_scratch_warmup_256.log

# python ./hubmap/experiments/TransResUNet/train.py\
#     --name resnext50_32x4d_scratch_warmup_256 \
#     --epochs 20 \
#     --backbone resnext50_32x4d \
#     --model TransResUNet \
#     --loss DiceBCELoss | tee ./logs/final/scratch/resnext50_32x4d_scratch_warmup_256.log

# python ./hubmap/experiments/TransResUNet/train.py\
#     --name resnext101_32x8d_scratch_warmup_256 \
#     --epochs 20 \
#     --backbone resnext101_32x8d \
#     --model TransResUNet \
#     --loss DiceBCELoss | tee ./logs/final/scratch/resnext101_32x8d_scratch_warmup_256.log

# python ./hubmap/experiments/TransResUNet/train.py\
#     --name wide_resnet50_2_scratch_warmup_256 \
#     --epochs 20 \
#     --backbone wide_resnet50_2 \
#     --model TransResUNet \
#     --loss DiceBCELoss | tee ./logs/final/scratch/wide_resnet50_2_scratch_warmup_256.log

python ./hubmap/experiments/TransResUNet/train.py\
    --name wide_resnet101_2_scratch_warmup_256 \
    --epochs 20 \
    --backbone wide_resnet101_2 \
    --model TransResUNet \
    --loss DiceBCELoss | tee ./logs/final/scratch/wide_resnet101_2_scratch_warmup_256.log
