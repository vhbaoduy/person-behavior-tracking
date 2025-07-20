#!/bin/bash
#SBATCH --job-name=VideoMAE_FT           # Job name
#SBATCH --partition=defq                 # Partition name
#SBATCH --ntasks=1                       # Number of tasks (1 process)
#SBATCH --gres=gpu:1                     # Number of GPUs
#SBATCH --cpus-per-task=14               # Number of CPUs per task
#SBATCH --output=logs/%x_%j.out          # STDOUT (%x=job name, %j=job ID)
#SBATCH --error=logs/%x_%j.err           # STDERR
#SBATCH --time=48:00:00                  # Max run time (hh:mm:ss)
#SBATCH --signal=USR1@60                 # For graceful termination

set -x  # Print each command

# ------------------- ENVIRONMENT SETUP -------------------
export OMP_NUM_THREADS=1
export MASTER_PORT=$((12000 + RANDOM % 20000))

# Activate your conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate paddle_nhan

# CUDA config (make sure this path is correct and nvcc is installed)
export CUDA_HOME=/home/clc_hcmus/miniconda3/envs/deepspeed_env
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Make sure nvcc exists
if ! command -v nvcc &> /dev/null; then
    echo "nvcc not found. Please check your CUDA installation at $CUDA_HOME"
    exit 1
fi


# ------------------- TRAINING CONFIG -------------------
JOB_NAME="VideoMAE_FT"
OUTPUT_DIR='./work_dir/vit_g_hybrid_pt_1200e_k710_ft'
DATA_PATH='./data/'
MODEL_PATH='./vit_g_hybrid_pt_1200e_k710_ft.pth'

python run_class_finetuning.py \
    --model vit_giant_patch14_224 \
    --data_set Kinetics-710 \
    --nb_classes 710 \
    --data_path ${DATA_PATH} \
    --finetune ${MODEL_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 3 \
    --input_size 224 \
    --short_side_size 224 \
    --save_ckpt_freq 10 \
    --num_frames 16 \
    --sampling_rate 4 \
    --num_sample 2 \
    --num_workers 10 \
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
    --dist_eval --enable_deepspeed
