#!/bin/sh
#SBATCH -p gpu-v100
#SBATCH --job-name=LD_Inference
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=12
#SBATCH -t 1-00:00:00
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
## #SBATCH --gres=gpu:p100:3
#SBATCH -o %x-%j.out

COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`
GRES=`scontrol show hostnames "$CUDA_VISIBLE_DEVICES" | wc -l`

torchrun --nproc_per_node=$GRES --master_port=23456 --nnodes=$COUNT_NODE /home/rbasiri/MyCode/Diffusion/latent-diffusion/Inference.py
# accelerate launch  /home/rbasiri/MyCode/Diffusion/latent-diffusion/Inference.py