#!/bin/bash

TESTPATH="/home/admin/workspace/project/datasets/mvsnet_dtu/dtu_test/dtu/"
TESTLIST="lists/dtu/test.txt"
OUTDIR="./outputs"

gpuid=$1
echo "start evaluation on DTU benchmark."
echo "resolution: 1152 x 864"
echo "utilizing default settings"
CKPT_FILE="./pretrained/pretrained.ckpt"
echo "${CKPT_FILE}"

# depth fusion with gipuma
CUDA_VISIBLE_DEVICES=$gpuid \
python test.py \
    --dataset=dtu_yao_eval --batch_size=1 \
    --testpath=$TESTPATH  --testlist=$TESTLIST \
    --loadckpt $CKPT_FILE --outdir $OUTDIR  \
    --interval_scale 1.06  --filter_method gipuma \
    --num_consistent 3 --prob_threshold 0.8 --disp_threshold 0.3

