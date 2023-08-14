#!/bin/bash

mkdir -p ./logs/pretrained

echo "PRETRAINED"
for file in hubmap/experiments/TransResUNet/pretrained/*; do
    log_file=${file##*/}
    log_file="${log_file%.*}"
    log_file=./logs/pretrained/"${log_file}".log
    python $file | tee $log_file
done

mkdir -p ./logs/scratch

echo "SCRATCH"
for file in hubmap/experiments/TransResUNet/pretrained/*; do
    log_file=${file##*/}
    log_file="${log_file%.*}"
    log_file=./logs/scratch/"${log_file}".log
    python $file | tee $log_file
done
