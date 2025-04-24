#!/bin/bash
#SBATCH --job-name=less_step2_train_grads  # Job name
#SBATCH --output=less_step2_train_grads_%A_%a.out # Standard output log (%A = Job ID, %a = Task ID)
#SBATCH --error=less_step2_train_grads_%A_%a.err  # Standard error log
#SBATCH --partition=gpu                  # Use the same GPU partition as warmup
#SBATCH --constraint=gpu80               # Use the same constraint as warmup
#SBATCH --nodes=1                        # Request one node
#SBATCH --gres=gpu:1                     # Request 1 GPU per task
#SBATCH --cpus-per-task=4                # Match warmup CPU request
#SBATCH --mem=64G                        # Match warmup memory request
#SBATCH --time=0-04:00:00                # Adjust time estimate (e.g., 4 hours per checkpoint) - MODIFY AS NEEDED
#SBATCH --array=0-3                      # Creates 4 tasks (0, 1, 2, 3) corresponding to checkpoints

# --- REDIRECT CACHES TO SCRATCH ---
export PIP_CACHE_DIR="/scratch/network/pw5115/.cache/pip"
export CONDA_PKGS_DIRS="/scratch/network/pw5115/.conda/pkgs"
export HF_HOME="/scratch/network/pw5115/.cache/huggingface"
export MPLCONFIGDIR="/scratch/network/pw5115/.config/matplotlib"
# Add others specific to your workflow (e.g., Transformers, Datasets)

# Create directories if they don't exist (conda might handle its own)
mkdir -p $PIP_CACHE_DIR
mkdir -p $(dirname $CONDA_PKGS_DIRS) # Make parent dir for conda
mkdir -p $HF_HOME
mkdir -p $MPLCONFIGDIR

echo "Cache directories redirected:"
echo "PIP_CACHE_DIR=$PIP_CACHE_DIR"
echo "CONDA_PKGS_DIRS=$CONDA_PKGS_DIRS"
echo "HF_HOME=$HF_HOME"
echo "MPLCONFIGDIR=$MPLCONFIGDIR"
# --- END CACHE REDIRECTION ---

echo "Job running on node: $(hostname)"
echo "SLURM_JOB_GPUS: $SLURM_JOB_GPUS"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# --- Environment Setup ---
module purge # Start clean

echo "Loading build modules..."
# module load gcc/11.4 # Example: Load a reasonably modern GCC (CHECK ADROIT DOCS/module avail)
module load cudatoolkit/12.2 # <<<--- CORRECTED VERSION based on avail and torch req.
echo "Modules loaded."

module load anaconda3/2024.10 # Load the same Anaconda module
conda activate less      # Activate the same Conda environment

export WANDB_MODE=offline   # Set WandB mode
echo "WANDB_MODE set to: $WANDB_MODE"

# --- Configuration ---
CODE_DIR="/scratch/network/pw5115/my_less_project/implicit-ins-improved/LESS" # Your project's code directory
CHECKPOINTS=(2468 4936 7405 9872) # Array of checkpoint numbers from warmup
CKPT_INDEX=$SLURM_ARRAY_TASK_ID
CKPT=${CHECKPOINTS[$CKPT_INDEX]}

WARMUP_JOB_NAME="llama2-7b-p0.05-lora-seed3" # Derived from your run_warmup.sh
TRAINING_DATA_NAME="openmathinstruct1"
# Construct full path based on CODE_DIR and relative data path from run_warmup.sh
TRAINING_DATA_FILE="${CODE_DIR}/data/train/processed/openmathinstruct1/openmathinstruct1_data.jsonl"
GRADIENT_TYPE="adam"
MODEL_PATH="${CODE_DIR}/../out/${WARMUP_JOB_NAME}/checkpoint-${CKPT}"
OUTPUT_PATH="${CODE_DIR}/grads/${WARMUP_JOB_NAME}/${TRAINING_DATA_NAME}-ckpt${CKPT}-${GRADIENT_TYPE}"
DIMS="8192" # Projection dimension

# --- Sanity Checks ---
echo "Current directory: $(pwd)" # Should be home directory initially
echo "Code Directory: $CODE_DIR"
echo "Using Checkpoint: $CKPT"
echo "Model Path: $MODEL_PATH"
echo "Output Path: $OUTPUT_PATH"
echo "Training Data File: $TRAINING_DATA_FILE"

if [ ! -d "$MODEL_PATH" ]; then
  echo "Error: Model checkpoint directory not found: $MODEL_PATH"
  exit 1
fi
if [ ! -f "$TRAINING_DATA_FILE" ]; then
  echo "Error: Training data file not found: $TRAINING_DATA_FILE"
  exit 1
fi

# --- Execute Gradient Calculation ---
echo "Navigating to code directory: ${CODE_DIR}"
cd "${CODE_DIR}"
echo "Current directory: $(pwd)"

SCRIPT_TO_RUN="./less/scripts/get_info/grad/get_train_lora_grads.sh"

echo "Starting gradient calculation for training data (Step 2) for Checkpoint ${CKPT}..."
# Ensure the script has execute permissions
chmod +x "${SCRIPT_TO_RUN}" 

"${SCRIPT_TO_RUN}" \
    "$TRAINING_DATA_FILE" \
    "$MODEL_PATH" \
    "$OUTPUT_PATH" \
    "$DIMS" \
    "$GRADIENT_TYPE"

EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
  echo "Error: get_train_lora_grads.sh failed with exit code $EXIT_CODE for checkpoint $CKPT."
  exit $EXIT_CODE
fi

echo "Finished Step 2 for Checkpoint: $CKPT."
echo "Wrapper script finished with status $?. Check Slurm output files."
# --- End of Job ---