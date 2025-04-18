#!/bin/bash
#SBATCH --job-name=less_warmup    # Job name for identification
#SBATCH --output=less_warmup_%j.out # Standard output log file (%j is the job ID)
#SBATCH --error=less_warmup_%j.err  # Standard error log file (%j is the job ID)
#SBATCH --partition=gpu           # Specify the GPU partition (IMPORTANT: Verify Adroit's GPU partition name, e.g., 'gpu', 'a100')
#SBATCH --constraint=gpu80
#SBATCH --nodes=1                 # Explicitly request one node
#SBATCH --gres=gpu:1              # Request 1 GPU (Llama 7B LoRA likely needs at least 1)
#SBATCH --cpus-per-task=4         # Request CPU cores (adjust based on potential data loading needs)
#SBATCH --mem=64G                 # Request RAM (e.g., 64GB; monitor usage and adjust if needed)
#SBATCH --time=0-08:00:00         # Maximum runtime (e.g., 8 hours; adjust based on estimate for 5% data)

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

# Execute your run_warmup script
# Make sure run_warmup.sh has execute permissions (chmod +x run_warmup.sh)
chmod +x run_warmup.sh
./run_warmup.sh

echo "Wrapper script finished with status $?. Check Slurm output files."

# --- End of Job ---