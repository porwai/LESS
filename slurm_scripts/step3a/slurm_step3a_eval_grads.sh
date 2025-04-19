#!/bin/bash
#SBATCH --job-name=less_step3a_eval_grads  # Job name
#SBATCH --output=less_step3a_eval_grads_%A_%a.out # Standard output log
#SBATCH --error=less_step3a_eval_grads_%A_%a.err  # Standard error log
#SBATCH --partition=gpu                  # Use the same GPU partition
#SBATCH --constraint=gpu80               # Use the same constraint
#SBATCH --nodes=1
#SBATCH --gres=gpu:1                     # Request 1 GPU per task
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=0-01:00:00                # Adjust time estimate (e.g., 1 hour per checkpoint) - MODIFY AS NEEDED
#SBATCH --array=0-3                      # Match the array range from Step 2

echo "Job running on node: $(hostname)"
echo "SLURM_JOB_GPUS: $SLURM_JOB_GPUS"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# --- Environment Setup ---
module purge
module load anaconda3/2024.10
conda activate LESS_env

export WANDB_MODE=offline
echo "WANDB_MODE set to: $WANDB_MODE"

# --- Configuration ---
CODE_DIR="/scratch/network/pw5115/my_less_project/implicit-ins-improved/LESS" # Your project's code directory
CHECKPOINTS=(105 211 317 420) # Match checkpoints from Step 2
CKPT_INDEX=$SLURM_ARRAY_TASK_ID
CKPT=${CHECKPOINTS[$CKPT_INDEX]}

WARMUP_JOB_NAME="llama2-7b-p0.05-lora-seed3" # Your warmup job identifier
TASK="alpaca_eval" # Your target task name
MODEL_PATH="${CODE_DIR}/out/${WARMUP_JOB_NAME}/checkpoint-${CKPT}"
OUTPUT_PATH="${CODE_DIR}/grads/${WARMUP_JOB_NAME}/${TASK}-ckpt${CKPT}-sgd" # Note '-sgd' suffix
# Base data dir where get_validation_dataset.py expects AlpacaEval data (relative to CODE_DIR)
DATA_DIR="${CODE_DIR}/data"
DIMS="4096 8192" # Projection dimensions

# --- Sanity Checks ---
echo "Current directory: $(pwd)" # Should be home directory initially
echo "Code Directory: $CODE_DIR"
echo "Using Checkpoint: $CKPT for Task: $TASK"
echo "Model Path: $MODEL_PATH"
echo "Output Path: $OUTPUT_PATH"
echo "Data Dir (for validation loader): $DATA_DIR"

if [ ! -d "$MODEL_PATH" ]; then
  echo "Error: Model checkpoint directory not found: $MODEL_PATH"
  exit 1
fi
# We assume the validation data loader script knows how to find data within DATA_DIR for the given TASK

# --- Execute AlpacaEval Gradient Calculation ---
echo "Navigating to code directory: ${CODE_DIR}"
cd "${CODE_DIR}"
echo "Current directory: $(pwd)"

echo "Starting gradient calculation for validation data (Step 3A) for Checkpoint ${CKPT}..."
# Ensure the script has execute permissions
chmod +x ./less/scripts/get_info/get_eval_lora_grads.sh

./less/scripts/get_info/get_eval_lora_grads.sh \
    "$TASK" \
    "$DATA_DIR" \
    "$MODEL_PATH" \
    "$OUTPUT_PATH" \
    "$DIMS"

EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
  echo "Error: get_eval_lora_grads.sh failed with exit code $EXIT_CODE for checkpoint $CKPT."
  exit $EXIT_CODE
fi

echo "Finished Step 3A for Checkpoint: $CKPT, Task: $TASK."
echo "Wrapper script finished with status $?. Check Slurm output files."
# --- End of Job ---