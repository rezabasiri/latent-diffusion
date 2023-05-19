#!/bin/sh
#SBATCH -p gpu-v100
#SBATCH --job-name=LD_Transfer
#SBATCH --nodes=3
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=12
#SBATCH -t 1-00:00:00
#SBATCH --mem=128G
#SBATCH --gres=gpu:2
## #SBATCH --gres=gpu:p100:3
#SBATCH -o %x-%j.out

# DATA_DIR="/home/rbasiri/Dataset/GAN/train_foot/train/"



echo myuser=`whoami`
echo COUNT_NODE=$COUNT_NODE
echo LD_LIBRARY_PATH = $LD_LIBRARY_PATH
echo PATH = $PATH
echo which mpicc `which mpicc`
echo HOSTNAMES = $HOSTNAMES
echo hostname = `hostname`
echo MASTER_ADDR= $MASTER_ADDR
echo MASTER_PORT= $MASTER_PORT

H=`hostname`
THEID=`echo -e $HOSTNAMES  | python3 -c "import sys;[sys.stdout.write(str(i)) for i,line in enumerate(next(sys.stdin).split(' ')) if line.strip() == '$H'.strip()]"`
echo THEID=$THEID


NUM_NODES=1
GPUS=1
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
# torchrun --nproc_per_node=$GPUS --master_port=23456 --nnodes=$NUM_NODES /home/rbasiri/MyCode/Diffusion/latent-diffusion/Training.py
accelerate launch  --num_processes $(( 2 * $COUNT_NODE )) --num_machines $COUNT_NODE --multi_gpu --mixed_precision fp16 --machine_rank $THEID --main_process_ip $MASTER_ADDR --main_process_port $MASTER_PORT /home/rbasiri/MyCode/Diffusion/latent-diffusion/Training.py