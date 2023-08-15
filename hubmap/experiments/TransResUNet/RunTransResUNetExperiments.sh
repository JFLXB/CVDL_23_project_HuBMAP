#!/bin/bash

echo "RUNNING FOR 256"
./hubmap/experiments/TransResUNet/RunTransResUNetExperiments_256.sh

echo "RUNNING FOR 512"
./hubmap/experiments/TransResUNet/RunTransResUNetExperiments_512.sh
