MVS_TRAINING="/home/admin/workspace/project/datasets/mvsnet_dtu/mvs_training/dtu"

DATE=`date +%Y-%m-%d`
LOG_DIR="./checkpoints/$1-$DATE"
if [ ! -d $LOG_DIR ]; then
    mkdir -p $LOG_DIR
fi

if [ "$1" == "unsup" ]; then  
    # baseline unsupervised training
    echo "unsupervised training!"

    python -u train_unsup_baseline.py \
        --gpu_device "0,1,2,3" \
        --dataset "dtu_yao" \
        --batch_size 4 \
        --trainpath $MVS_TRAINING \
        --trainlist "lists/dtu/train.txt" \
        --testlist "lists/dtu/test.txt" \
        --numdepth 192 \
        --logdir $LOG_DIR \
        --refine False \
        --lr 0.001 \
        --epochs 10 \
        --lrepochs "2,4,6,8:2" \
        --summary_freq 500 \
	    --val_freq 5000 \
        | tee -a $LOG_DIR/log.txt

elif [ "$1" == "selfsup_pretrain" ]; then
    # self-supervised pretraining (Section 3.2 in the paper)
    echo "unsupervised training with flow consistency and data augmentation!"

    python -u train_selfsup_pretrain.py \
        --gpu_device "0,1,2,3" \
        --dataset "dtu_yao_flow" \
        --batch_size 4 \
        --trainpath $MVS_TRAINING \
        --trainlist "lists/dtu/train.txt" \
        --testlist "lists/dtu/test.txt" \
        --numdepth 192 \
        --logdir $LOG_DIR \
        --refine False \
        --lr 0.001 \
        --epochs 10 \
        --lrepochs "2,4,6,8:2" \
        --summary_freq 500 \
	    --val_freq 5000 \
        --w_aug 0.01 \
        --w_flow 8.0 \
        | tee -a $LOG_DIR/log.txt

elif [ "$1" == "pselbl_postrain" ]; then
    # pseudo-label post-training (Section 3.3 in the paper)
    echo "psedo label post training."
    # PRETRAINED_WEIGHT="./checkpoints/selfsup_pretrain/model_00065000.ckpt"
    PRETRAINED_WEIGHT="./pretrained/selfsup_pretrain/model_00065000.ckpt"

    python -u train_pselbl_postrain.py \
        --gpu_device "0,1,2,3" \
        --dataset "dtu_yao_uncertainty" \
        --batch_size 4 \
        --trainpath $MVS_TRAINING \
        --trainlist "lists/dtu/train.txt" \
        --testlist "lists/dtu/test.txt" \
        --numdepth 192 \
        --logdir $LOG_DIR \
        --refine False \
        --lr 0.001 \
        --epochs 8 \
        --lrepochs "2,4,6,8:2" \
        --summary_freq 500 \
	    --val_freq 5000 \
        --w_aug 0.01 \
        --w_flow 8.0 \
        --loadckpt $PRETRAINED_WEIGHT \
        | tee -a $LOG_DIR/log.txt

fi