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

HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=23456
COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`
GRES=`scontrol show hostnames "$CUDA_VISIBLE_DEVICES" | wc -l`

echo myuser=`whoami`
echo COUNT_NODE=$COUNT_NODE
echo COUNT_GPU=$GRES
# echo LD_LIBRARY_PATH = $LD_LIBRARY_PATH
# echo PATH = $PATH
echo which mpicc `which mpicc`
echo HOSTNAMES = $HOSTNAMES
echo hostname = `hostname`
echo MASTER_ADDR= $MASTER_ADDR
echo MASTER_PORT= $MASTER_PORT

H=`hostname`
THEID=`echo -e $HOSTNAMES  | python3 -c "import sys;[sys.stdout.write(str(i)) for i,line in enumerate(next(sys.stdin).split(' ')) if line.strip() == '$H'.strip()]"`
echo THEID=$THEID

# torchrun --nproc_per_node=$GRES --master_port=23456 --nnodes=$COUNT_NODE /home/rbasiri/MyCode/Diffusion/latent-diffusion/Inference.py
accelerate launch --num_processes $(( $GRES * $COUNT_NODE )) --num_machines $COUNT_NODE --machine_rank $THEID --main_process_ip $MASTER_ADDR --main_process_port $MASTER_PORT /home/rbasiri/MyCode/Diffusion/latent-diffusion/Inference.py