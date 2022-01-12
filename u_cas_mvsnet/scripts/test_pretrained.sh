#!/bin/bash

TESTPATH="/home/admin/workspace/project/datasets/mvsnet_dtu/dtu_test/dtu/"
TESTLIST="lists/dtu/test.txt"
OUTDIR="./outputs_pretrained"
# CKPT_FILE="./checkpoints/$1/model_000015.ckpt"
CKPT_FILE="./pretrained/pretrained.ckpt"
RESOLUTION_MODE=$1

if [ "$RESOLUTION_MODE" == "high" ]; then
  echo "high resolution: 1600 x 1200"

  # depth fusion with gipuma
  CUDA_VISIBLE_DEVICES=1 \
  python -u test_pretrained.py \
      --dataset=general_eval --batch_size=1 \
      --testpath=$TESTPATH  --testlist=$TESTLIST \
      --loadckpt $CKPT_FILE --outdir $OUTDIR  --interval_scale 1.06  --filter_method gipuma \
      --max_h 1200 --max_w 1600 \
      --num_consistent 3 --prob_threshold 0.4 --disp_threshold 0.25

elif [ "$RESOLUTION_MODE" == "medium" ]; then
  echo "medium resolution: 1152 x 864"
  echo "utilizing default settings"

  # depth fusion with gipuma
  CUDA_VISIBLE_DEVICES=1 \
  python -u test_pretrained.py \
      --dataset=general_eval --batch_size=1 \
      --testpath=$TESTPATH  --testlist=$TESTLIST \
      --loadckpt $CKPT_FILE --outdir $OUTDIR  --interval_scale 1.06  --filter_method gipuma \
      --num_consistent 3 --prob_threshold 0.4 --disp_threshold 0.25

fi
