#!/usr/bin/env bash
set -x  # print the commands

export MASTER_PORT=$((12000 + $RANDOM % 20000))  # Randomly set master_port to avoid port conflicts
export OMP_NUM_THREADS=1  # Control the number of threads

OUTPUT_DIR='./work_dir/'  # Your output folder for deepspeed config file, logs and checkpoints
DATA_PATH='./data/'  # The data list folder. the folder has three files: train.csv, val.csv, test.csv
MODEL_PATH='./vit_b_k710_dl_from_giant.pth'  # Model for initializing parameters

JOB_NAME=$1  # the job name of the slurm task
PARTITION=defq  # Name of the partition

GPUS=1
GPUS_PER_NODE=1
CPUS_PER_TASK=14
SRUN_ARGS=${SRUN_ARGS:-""}  # Other slurm task args
PY_ARGS=${@:2}  # Other training args

python run_class_finetuning.py \
    --model vit_base_patch16_224 \
    --data_set Kinetics-400 \
    --nb_classes 10 \
    --data_path ${DATA_PATH} \
    --finetune ${MODEL_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 8 \
    --input_size 224 \
    --short_side_size 224 \
    --save_ckpt_freq 10 \
    --num_frames 16 \
    --sampling_rate 4 \
    --num_sample 2 \
    --num_workers 8 \
    --opt adamw \
    --lr 1e-3 \
    --drop_path 0.3 \
    --clip_grad 5.0 \
    --layer_decay 0.9 \
    --opt_betas 0.9 0.999 \
    --weight_decay 0.1 \
    --warmup_epochs 5 \
    --epochs 35 \
    --test_num_segment 5 \
    --test_num_crop 3 \
    --dist_eval --enable_deepspeed >${OUTPUT_DIR}/res.out 2>${OUTPUT_DIR}/res.err
