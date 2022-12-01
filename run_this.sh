#!/bin/sh
#SBATCH --job-name=rdetr
#SBATCH -o /work/pi_hzhang2_umass_edu/snagabhushan_umass_edu/rdetr/logs/long_train_input.txt
#SBATCH --time=20:00:00
#SBATCH -c 1 # Cores
#SBATCH --mem=64GB  # Requested Memory
#SBATCH -p gpu  # Partition
#SBATCH -G 2 # Number of GPUs

echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR
echo $(nvidia-smi)

eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate py39

cd /work/pi_hzhang2_umass_edu/snagabhushan_umass_edu/rdetr

python -m torch.distributed.launch --nproc_per_node=2 --use_env main.py --dataset_config configs/lvis.json --output-dir runs --ema > logs/run.txt