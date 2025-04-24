#!/bin/bash
#SBATCH --job-name=less_warmup_multi_gpu # Changed job name slightly
#SBATCH --output=less_warmup_%j.out
#SBATCH --error=less_warmup_%j.err
#SBATCH --partition=gpu                   # Keep partition (verify 'gpu' is correct for multi-A100 nodes on Adroit)
#SBATCH --constraint=gpu80                # Keep constraint for A100 80GB GPUs
#SBATCH --nodes=1                         # Request one node (typical for FSDP unless configured for multi-node)
#SBATCH --gres=gpu:2                      # Request 4 GPUs (Adjust to 2, 4, or 8 based on Adroit node availability and code scaling)
#SBATCH --cpus-per-task=8                # Request CPU cores (e.g., 4 cores per GPU requested)
#SBATCH --mem=128G                        # Request RAM (e.g., 64GB per GPU requested - adjust based on monitoring)
#SBATCH --time=0-16:00:00                 # Maximum runtime (adjust based on estimated speedup - 16h might be safe)

echo "Job running on node: $(hostname)"
echo "SLURM_JOB_GPUS: $SLURM_JOB_GPUS" # See which GPU IDs Slurm allocated
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES" # See which GPUs CUDA apps will see

# --- Environment Setup ---
module purge # Start with a clean environment
module load anaconda3/2024.10
# module load cuda/<version> cudnn/<version> # May be needed depending on PyTorch install

# Activate your Conda environment
conda activate LESS_env # Use the name of the environment you created

# --- Job Execution ---
# Navigate to the directory where your LESS code repository is located
cd /scratch/network/pw5115/my_less_project/implicit-ins-improved/LESS

echo "Current directory: $(pwd)"
echo "Starting run_warmup script..."

export WANDB_MODE=offline
echo "WANDB_MODE set to: $WANDB_MODE"

# Execute your run_warmup script
# Make sure run_warmup.sh has execute permissions (chmod +x run_warmup.sh)
chmod +x run_warmup.sh
./run_warmup.sh

echo "Wrapper script finished with status $?. Check Slurm output files."

# --- End of Job ---