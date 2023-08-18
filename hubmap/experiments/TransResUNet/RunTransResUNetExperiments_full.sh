#!/bin/bash

# echo "RUNNING FOR 256"
# ./hubmap/experiments/TransResUNet/RunTransResUNetExperiments_256.sh

# echo "RUNNING FOR 512"
# ./hubmap/experiments/TransResUNet/RunTransResUNetExperiments_512.sh

# echo "RUNNING FINAL MODEL FOR 256 IMAGE SIZES: RestNet-152"
# python ./hubmap/experiments/TransResUNet/train.py\
#     --name resnet152_x256 \
#     --backbone resnet152 \
#     --pretrained \
#     --model TransResUNet \
#     --epochs 200 \
#     --loss DiceBCELoss \
#     --use-lr-scheduler \
#     --lrs-patience 5 \
#     --use-early-stopping \
#     --continue-training \
#     --from-checkpoint TransResUNet/resnet152_pretrained_warmup_256.pt \
#     --es-patience 50 | tee ./logs/final/eval/resnet152_x256.log


# echo "RUNNING FINAL MODEL FOR 256 IMAGE SIZES: ResNeXt-50 32x4d"
# python ./hubmap/experiments/TransResUNet/train.py\
#     --name resnext50_32x4d_x256 \
#     --backbone resnext50_32x4d \
#     --pretrained \
#     --model TransResUNet \
#     --epochs 200 \
#     --loss DiceBCELoss \
#     --use-lr-scheduler \
#     --lrs-patience 5 \
#     --use-early-stopping \
#     --continue-training \
#     --from-checkpoint TransResUNet/resnext50_32x4d_pretrained_warmup_256.pt \
#     --es-patience 50 | tee ./logs/final/eval/resnext50_32x4d_x256.log


# echo "RUNNING FINAL MODEL FOR 256 IMAGE SIZES: ResNeXt-101 32x8d"
# python ./hubmap/experiments/TransResUNet/train.py\
#     --name resnext101_32x8d_x256 \
#     --backbone resnext101_32x8d \
#     --pretrained \
#     --model TransResUNet \
#     --epochs 200 \
#     --loss DiceBCELoss \
#     --use-lr-scheduler \
#     --lrs-patience 5 \
#     --use-early-stopping \
#     --continue-training \
#     --from-checkpoint TransResUNet/resnext101_32x8d_pretrained_warmup_256.pt \
#     --es-patience 50 | tee ./logs/final/eval/resnext101_32x8d_x256.log


# echo "RUNNING FINAL MODEL FOR 256 IMAGE SIZES: Wide RestNet-50-2"
# python ./hubmap/experiments/TransResUNet/train.py\
#     --name wide_resnet50_2_x256 \
#     --backbone wide_resnet50_2 \
#     --pretrained \
#     --model TransResUNet \
#     --epochs 200 \
#     --loss DiceBCELoss \
#     --use-lr-scheduler \
#     --lrs-patience 5 \
#     --use-early-stopping \
#     --continue-training \
#     --from-checkpoint TransResUNet/wide_resnet50_2_pretrained_warmup_256.pt \
#     --es-patience 50 | tee ./logs/final/eval/wide_resnet50_2_x256.log

# echo "RUNNING FINAL MODEL FOR 256 IMAGE SIZES: Wide RestNet-101-2"
# python ./hubmap/experiments/TransResUNet/train.py\
#     --name wide_resnet101_2_x256 \
#     --backbone wide_resnet101_2 \
#     --pretrained \
#     --model TransResUNet \
#     --epochs 200 \
#     --loss DiceBCELoss \
#     --use-lr-scheduler \
#     --lrs-patience 5 \
#     --use-early-stopping \
#     --continue-training \
#     --from-checkpoint TransResUNet/wide_resnet101_2_pretrained_warmup_256.pt \
#     --es-patience 50 | tee ./logs/final/eval/wide_resnet101_2_x256.log

# ############################################################################################################
# ############################################################################################################
# ############################################################################################################
# ############################################################################################################
# ############################################################################################################
# ############################################################################################################

# echo "NOW 512"

# echo "RUNNING FINAL MODEL FOR 512 IMAGE SIZES: RestNet-152"
# python ./hubmap/experiments/TransResUNet/train.py\
#     --name resnet152_x512 \
#     --backbone resnet152 \
#     --pretrained \
#     --model TransResUNet512 \
#     --epochs 200 \
#     --loss DiceBCELoss \
#     --use-lr-scheduler \
#     --lrs-patience 5 \
#     --use-early-stopping \
#     --continue-training \
#     --from-checkpoint TransResUNet/resnet152_pretrained_warmup_512.pt \
#     --es-patience 50 | tee ./logs/final/eval/resnet152_x512.log


# echo "RUNNING FINAL MODEL FOR 512 IMAGE SIZES: RestNet-101"
# python ./hubmap/experiments/TransResUNet/train.py\
#     --name resnet101_x512 \
#     --backbone resnet101 \
#     --pretrained \
#     --model TransResUNet512 \
#     --epochs 200 \
#     --loss DiceBCELoss \
#     --use-lr-scheduler \
#     --lrs-patience 5 \
#     --use-early-stopping \
#     --continue-training \
#     --from-checkpoint TransResUNet/resnet101_pretrained_warmup_512.pt \
#     --es-patience 50 | tee ./logs/final/eval/resnet101_x512.log


# echo "RUNNING FINAL MODEL FOR 512 IMAGE SIZES: ResNeXt-50 32x4d"
# python ./hubmap/experiments/TransResUNet/train.py\
#     --name resnext50_32x4d_x512 \
#     --backbone resnext50_32x4d \
#     --pretrained \
#     --model TransResUNet512 \
#     --epochs 200 \
#     --loss DiceBCELoss \
#     --use-lr-scheduler \
#     --lrs-patience 5 \
#     --use-early-stopping \
#     --continue-training \
#     --from-checkpoint TransResUNet/resnext50_32x4d_pretrained_warmup_512.pt \
#     --es-patience 50 | tee ./logs/final/eval/resnext50_32x4d_x512.log


# echo "RUNNING FINAL MODEL FOR 512 IMAGE SIZES: ResNeXt-101 32x8d"
# python ./hubmap/experiments/TransResUNet/train.py\
#     --name resnext101_32x8d_x512 \
#     --backbone resnext101_32x8d \
#     --pretrained \
#     --model TransResUNet512 \
#     --epochs 200 \
#     --loss DiceBCELoss \
#     --use-lr-scheduler \
#     --lrs-patience 5 \
#     --use-early-stopping \
#     --continue-training \
#     --from-checkpoint TransResUNet/resnext101_32x8d_pretrained_warmup_512.pt \
#     --es-patience 50 | tee ./logs/final/eval/resnext101_32x8d_x512.log


# echo "RUNNING FINAL MODEL FOR 512 IMAGE SIZES: Wide RestNet-50-2"
# python ./hubmap/experiments/TransResUNet/train.py\
#     --name wide_resnet50_2_x512 \
#     --backbone wide_resnet50_2 \
#     --pretrained \
#     --model TransResUNet512 \
#     --epochs 200 \
#     --loss DiceBCELoss \
#     --use-lr-scheduler \
#     --lrs-patience 5 \
#     --use-early-stopping \
#     --continue-training \
#     --from-checkpoint TransResUNet/wide_resnet50_2_pretrained_warmup_512.pt \
#     --es-patience 50 | tee ./logs/final/eval/wide_resnet50_2_x512.log

# echo "RUNNING FINAL MODEL FOR 512 IMAGE SIZES: Wide RestNet-101-2"
# python ./hubmap/experiments/TransResUNet/train.py\
#     --name wide_resnet101_2_x512 \
#     --backbone wide_resnet101_2 \
#     --pretrained \
#     --model TransResUNet512 \
#     --epochs 200 \
#     --loss DiceBCELoss \
#     --use-lr-scheduler \
#     --lrs-patience 5 \
#     --use-early-stopping \
#     --continue-training \
#     --from-checkpoint TransResUNet/wide_resnet101_2_pretrained_warmup_512.pt \
#     --es-patience 50 | tee ./logs/final/eval/wide_resnet101_2_x512.log

############################################################################################################
############################################################################################################
############################################################################################################
############################################################################################################

# Channel Weighted DiceBCELoss
echo "Channel Weighted DiceBCELoss"

# echo "Warmup RUNNING Channel Weighted DiceBCELoss MODEL FOR 256 IMAGE SIZES: Wide RestNet-50-2"
# python ./hubmap/experiments/TransResUNet/train.py\
#     --name wide_resnet50_2_x256_channel_weighted_warmup \
#     --backbone wide_resnet50_2 \
#     --pretrained \
#     --model TransResUNet \
#     --epochs 20 \
#     --loss ChannelWeightedDiceBCELoss \
#     --weights 1.2753886168323578 1.2737381309347808 1.3285854321630461 0.12228782006981492 \
#     | tee ./logs/final/eval/wide_resnet50_2_x256_channel_weighted_warmup.log

echo "RUNNING Channel Weighted DiceBCELoss MODEL FOR 256 IMAGE SIZES: Wide RestNet-50-2"
python ./hubmap/experiments/TransResUNet/train.py\
    --name wide_resnet50_2_x256_channel_weighted \
    --backbone wide_resnet50_2 \
    --pretrained \
    --model TransResUNet \
    --epochs 200 \
    --loss ChannelWeightedDiceBCELoss \
    --weights 1.2753886168323578 1.2737381309347808 1.3285854321630461 0.12228782006981492 \
    --continue-training \
    --from-checkpoint TransResUNet/wide_resnet50_2_x256_channel_weighted_warmup.pt \
    --use-early-stopping \
    --es-patience 50 \
    --use-lr-scheduler \
    --lrs-patience 5 | tee ./logs/final/eval/wide_resnet50_2_x256_channel_weighted.log



# echo "Warmup RUNNING Channel Weighted DiceBCELoss MODEL FOR 512 IMAGE SIZES: ResNeXt-101 32x8d"
# python ./hubmap/experiments/TransResUNet/train.py \
#     --name resnext101_32x8d_x512_channel_weighted_warmup \
#     --backbone resnext101_32x8d \
#     --pretrained \
#     --model TransResUNet512 \
#     --epochs 20 \
#     --loss ChannelWeightedDiceBCELoss \
#     --weights 1.2753886168323578 1.2737381309347808 1.3285854321630461 0.12228782006981492 \
#     | tee ./logs/final/eval/resnext101_32x8d_x512_channel_weighted_warmup.log

echo "RUNNING Channel Weighted DiceBCELoss MODEL FOR 512 IMAGE SIZES: ResNeXt-101 32x8d"
python ./hubmap/experiments/TransResUNet/train.py\
    --name resnext101_32x8d_x512_channel_weighted \
    --backbone resnext101_32x8d \
    --pretrained \
    --model TransResUNet512 \
    --epochs 200 \
    --loss ChannelWeightedDiceBCELoss \
    --weights 1.2753886168323578 1.2737381309347808 1.3285854321630461 0.12228782006981492 \
    --continue-training \
    --from-checkpoint TransResUNet/resnext101_32x8d_x512_channel_weighted_warmup.pt \
    --use-early-stopping \
    --es-patience 50 \
    --use-lr-scheduler \
    --lrs-patience 5 | tee ./logs/final/eval/resnext101_32x8d_x512_channel_weighted.log



# echo "BACKUP FOR ResNet-152"

# echo "Warmup RUNNING Channel Weighted DiceBCELoss MODEL FOR 512 IMAGE SIZES: ResNet-152"
# python ./hubmap/experiments/TransResUNet/train.py \
#     --name resnet152_x512_channel_weighted_warmup \
#     --backbone resnet152 \
#     --pretrained \
#     --model TransResUNet512 \
#     --epochs 20 \
#     --loss ChannelWeightedDiceBCELoss \
#     --weights 1.2753886168323578 1.2737381309347808 1.3285854321630461 0.12228782006981492 \
#     | tee ./logs/final/eval/resnet152_x512_channel_weighted_warmup.log

# echo "RUNNING Channel Weighted DiceBCELoss MODEL FOR 512 IMAGE SIZES: ResNet-152"
# python ./hubmap/experiments/TransResUNet/train.py\
#     --name resnet152_x512_channel_weighted \
#     --backbone resnet152 \
#     --pretrained \
#     --model TransResUNet512 \
#     --epochs 200 \
#     --loss ChannelWeightedDiceBCELoss \
#     --weights 1.2753886168323578 1.2737381309347808 1.3285854321630461 0.12228782006981492 \
#     --continue-training \
#     --from-checkpoint TransResUNet/resnet152_x512_channel_weighted_warmup.pt \
#     --use-early-stopping \
#     --es-patience 50 \
#     --use-lr-scheduler \
#     --lrs-patience 5 | tee ./logs/final/eval/resnet152_x512_channel_weighted.log


