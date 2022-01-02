MVS_TRAINING="/home/admin/workspace/project/datasets/mvsnet_dtu/mvs_training/dtu"
OUTDIR="./uncertainty"
PRETRAINED_WEIGHT="./pretrained/selfsup_pretrain/model_00065000.ckpt"

echo "Utilizing pretrained weight: {$PRETRAINED_WEIGHT}"
echo "Generating pseudo label and uncertainty map."

python -u gen_pselbl.py \
    --gpu_device "0,1,2,3" \
    --dataset "dtu_yao" \
    --batch_size 4 \
    --trainpath $MVS_TRAINING \
    --trainlist "lists/dtu/train.txt" \
    --testlist "lists/dtu/test.txt" \
    --numdepth 192 \
    --logdir $OUTDIR \
    --refine False \
    --loadckpt $PRETRAINED_WEIGHT \
    --outdir $OUTDIR \
    --test_trials 20 

