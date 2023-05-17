#!/bin/sh
#SBATCH -p gpu-a100
#SBATCH --job-name=LDM256
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=12
#SBATCH -t 1-00:00:00
#SBATCH --mem=256G
#SBATCH --gres=gpu:1
## #SBATCH --gres=gpu:p100:3
#SBATCH -o %x-%j.out

# DATA_DIR="/home/rbasiri/Dataset/GAN/train_foot/train/"

NUM_NODES=1
GPUS=2
# DEPTH=24
# DSteps=2000
# IMG_SIZE=128
# BATCH_PER_GPU=12
# MODEL_NAME="vit_xl_patch2_32"
# EXP_NAME=layer8_Dsteps2000_640480_footbkrm_1stRun

# MODEL_BLOB="/mnt/external"
# if [ ! -d $MODEL_BLOB ]; then
#     MODEL_BLOB="/home/rbasiri/Dataset/saved_models/Diffusion/latent"
# fi

# OPENAI_LOGDIR="${MODEL_BLOB}/$EXP_NAME"

# mkdir -p $OPENAI_LOGDIR
# OPENAI_LOGDIR=$OPENAI_LOGDIR \
#     
torchrun --nproc_per_node=$GPUS --master_port=23456 --nnodes=$NUM_NODES /home/rbasiri/MyCode/Diffusion/latent-diffusion/Training.py