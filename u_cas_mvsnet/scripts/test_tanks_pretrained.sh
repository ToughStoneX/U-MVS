#!/bin/bash
SPLIT=$1  # intermediate / advanced
DATAPATH="/home/admin/workspace/project/datasets/patchmatchnet_data/TankandTemples/"
GPUID=$2
OUTDIR="./outputs_tanks"
# CKPT_FILE="./checkpoints/$3/model_000015.ckpt"
CKPT_FILE="./pretrained/pretrained.ckpt"


if [ "$SPLIT" == "intermediate" ]; then
    echo "tanksandtemples benchmark; split: $SPLIT"
    
    CUDA_VISIBLE_DEVICES=$GPUID \
    python test_tanks.py \
        --dataset "tanks" --split $SPLIT --batch_size 1 \
        --testpath $DATAPATH --loadckpt $CKPT_FILE \
        --outdir $OUTDIR --num_view 7 \
        --max_h 1056 --max_w 1920 \
        --geo_pixel_thres=1 --geo_depth_thres=0.01

elif [ "$SPLIT" == "advanced" ]; then
    echo "tanksandtemples benchmark; split: $SPLIT"

    CUDA_VISIBLE_DEVICES=$GPUID \
    python test_tanks.py \
        --dataset "tanks" --split $SPLIT --batch_size 1 \
        --testpath $DATAPATH --loadckpt $CKPT_FILE \
        --outdir $OUTDIR --num_view 7 \
        --max_h 1056 --max_w 1920 \
        --geo_pixel_thres=1 --geo_depth_thres=0.01

fi