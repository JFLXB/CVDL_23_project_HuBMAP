#!/bin/bash

# echo "RUNNING FOR 256"
# ./hubmap/experiments/TransResUNet/RunTransResUNetExperiments_256.sh

# echo "RUNNING FOR 512"
# ./hubmap/experiments/TransResUNet/RunTransResUNetExperiments_512.sh



# echo "RUNNING FINAL MODEL FOR 256 IMAGE SIZES: Wide RestNet-50-2"
# python ./hubmap/experiments/TransResUNet/train.py\
#     --name wide_resnet50_2_x256 \
#     --backbone wide_resnet50_2 \
#     --pretrained \
#     --model TransResUNet \
#     --epochs 500 \
#     --loss DiceBCELoss \
#     --use-lr-scheduler \
#     --lrs-patience 5 \
#     --use-early-stopping \
#     --continue-training \
#     --from-checkpoint TransResUNet/wide_resnet50_2_pretrained_warmup_256.pt \
#     --es-patience 50 | tee ./logs/final/eval/wide_resnet50_2_x256.log

echo "RUNNING FINAL MODEL FOR 512 IMAGE SIZES: Wide RestNet-50-2"
python ./hubmap/experiments/TransResUNet/train.py\
    --name wide_resnet50_2_x512 \
    --backbone wide_resnet50_2 \
    --pretrained \
    --model TransResUNet512 \
    --epochs 500 \
    --loss DiceBCELoss \
    --use-lr-scheduler \
    --lrs-patience 5 \
    --use-early-stopping \
    --continue-training \
    --from-checkpoint TransResUNet/wide_resnet50_2_pretrained_warmup_512.pt \
    --es-patience 50 | tee ./logs/final/eval/wide_resnet50_2_x512.log


# echo "RUNNING FINAL MODEL FOR 512 IMAGE SIZES: Wide RestNet-50-2"
# python ./hubmap/experiments/TransResUNet/train.py\
#     --name wide_resnet50_2_x256 \
#     --backbone wide_resnet50_2 \
#     --pretrained \
#     --model TransResUNet \
#     --epochs 500 \
#     --loss DiceBCELoss \
#     --use-lr-scheduler \
#     --lrs-patience 5 \
#     --use-early-stopping \
#     --es-patience 50 | tee ./logs/final/scratch/wide_resnet50_2_x256.log
