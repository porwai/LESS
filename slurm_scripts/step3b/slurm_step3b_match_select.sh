#!/bin/bash
#SBATCH --job-name=less_step3b_match_select # Job name
#SBATCH --output=less_step3b_match_select_%A.out # Standard output log
#SBATCH --error=less_step3b_match_select_%A.err  # Standard error log
#SBATCH --partition=gpu                  # Using GPU partition for consistency, maybe faster? Or use a CPU partition if preferred.
#SBATCH --constraint=gpu80               # Keeping constraint, remove if not needed/running on CPU partition
#SBATCH --nodes=1
#SBATCH --gres=gpu:1                     # Optional: Remove if using CPU partition
#SBATCH --cpus-per-task=8                # More CPUs might help matching
#SBATCH --mem=128G                       # More memory for matching
#SBATCH --time=0-02:00:00                # Adjust time estimate (e.g., 2 hours) - MODIFY AS NEEDED
#SBATCH --dependency=afterok:<JOB_ID_STEP2>:<JOB_ID_STEP3A> # IMPORTANT: Replace with actual Job IDs after submitting Step 2 and 3A

echo "Job running on node: $(hostname)"
# Only show GPU info if we requested one
if [ -n "$SLURM_JOB_GPUS" ]; then
  echo "SLURM_JOB_GPUS: $SLURM_JOB_GPUS"
  echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
fi

# --- Environment Setup ---
module purge
module load anaconda3/2024.10
# Load 'bc' if it's a separate module and not included in base environment
# module load bc
conda activate LESS_env

export WANDB_MODE=offline
echo "WANDB_MODE set to: $WANDB_MODE"

# --- Configuration ---
CODE_DIR="/scratch/network/pw5115/my_less_project/implicit-ins-improved/LESS" # Your project's code directory
DIM=8192 # Dimension used for gradients
WARMUP_JOB_NAME="llama2-7b-p0.05-lora-seed3" # Your warmup job identifier

# Template paths using {} placeholders, based within CODE_DIR
GRADIENT_PATH_TEMPLATE="${CODE_DIR}/grads/${WARMUP_JOB_NAME}/{}-ckpt{}-adam/dim${DIM}"
VALIDATION_GRADIENT_PATH_TEMPLATE="${CODE_DIR}/grads/${WARMUP_JOB_NAME}/{}-ckpt{}-sgd/dim${DIM}"

TRAIN_FILE_NAMES="openMathInstruct-1"
CKPTS_STR="105 211 317 420" # Space-separated string of checkpoints used
# Example weights (average LR per epoch). Adjust if needed or use equal weights e.g., "0.25 0.25 0.25 0.25"
CHECKPOINT_WEIGHTS="1.6877e-05 1.2859e-05 7.7030e-06 2.5616e-06"

TARGET_TASK_NAMES="alpaca_eval"
SELECTED_DATA_OUTPUT_PATH="${CODE_DIR}/selected_data_omi1_alpaca_eval" # Output directory within CODE_DIR
OMI1_TRAIN_FILE="${CODE_DIR}/data/train/processed/openMathInstruct-1/openMathInstruct-1_data.jsonl" # Full path

# --- Calculate Selection Percentage for 1000 examples ---
echo "Calculating selection percentage..."
if ! command -v bc &> /dev/null; then
    echo "Error: 'bc' command could not be found. Load the 'bc' module or install it."
    exit 1
fi
if [ ! -f "$OMI1_TRAIN_FILE" ]; then
    echo "Error: Training data file not found for counting: $OMI1_TRAIN_FILE"
    exit 1
fi
TOTAL_EXAMPLES=$(wc -l < "${OMI1_TRAIN_FILE}")
if [ "$TOTAL_EXAMPLES" -eq 0 ]; then
  echo "Error: Cannot determine total examples in ${OMI1_TRAIN_FILE}. It might be empty."
  exit 1
fi
# Use bc for floating point division with sufficient precision
SELECTION_PERCENTAGE=$(echo "scale=8; 1000 / $TOTAL_EXAMPLES" | bc)
if (( $(echo "$SELECTION_PERCENTAGE <= 0" | bc -l) )); then
    echo "Error: Calculated percentage is zero or negative ($SELECTION_PERCENTAGE). Check TOTAL_EXAMPLES ($TOTAL_EXAMPLES)."
    exit 1
fi
echo "Total examples in ${TRAIN_FILE_NAMES}: $TOTAL_EXAMPLES"
echo "Target selection count: 1000"
echo "Calculated selection percentage: $SELECTION_PERCENTAGE"

# --- Sanity Checks ---
echo "Current directory: $(pwd)" # Should be home directory initially
echo "Code Directory: $CODE_DIR"
echo "Output Path for selected data: $SELECTED_DATA_OUTPUT_PATH"
echo "Dependencies: $SLURM_JOB_DEPENDENCY"

# Basic check for gradient directories from the first checkpoint
FIRST_CKPT=$(echo $CKPTS_STR | awk '{print $1}')
EXPECTED_TRAIN_GRAD_DIR=$(echo $GRADIENT_PATH_TEMPLATE | sed "s/{}/${TRAIN_FILE_NAMES}/g" | sed "s/{}/${FIRST_CKPT}/g")
EXPECTED_EVAL_GRAD_DIR=$(echo $VALIDATION_GRADIENT_PATH_TEMPLATE | sed "s/{}/${TARGET_TASK_NAMES}/g" | sed "s/{}/${FIRST_CKPT}/g")
if [ ! -d "$EXPECTED_TRAIN_GRAD_DIR" ]; then    echo "Warning: Expected training gradient directory not found: $EXPECTED_TRAIN_GRAD_DIR"; fi
if [ ! -d "$EXPECTED_EVAL_GRAD_DIR" ]; then    echo "Warning: Expected validation gradient directory not found: $EXPECTED_EVAL_GRAD_DIR"; fi

# --- Execute Matching ---
echo "Navigating to code directory: ${CODE_DIR}"
cd "${CODE_DIR}"
echo "Current directory: $(pwd)"

echo "Starting matching script (matching.sh)..."
mkdir -p $SELECTED_DATA_OUTPUT_PATH # Ensure output directory exists
chmod +x ./less/scripts/data_selection/matching.sh

./less/scripts/data_selection/matching.sh \
    "$GRADIENT_PATH_TEMPLATE" \
    "$TRAIN_FILE_NAMES" \
    "$CKPTS_STR" \
    "$CHECKPOINT_WEIGHTS" \
    "$VALIDATION_GRADIENT_PATH_TEMPLATE" \
    "$TARGET_TASK_NAMES" \
    "$SELECTED_DATA_OUTPUT_PATH"

EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then  echo "Error: matching.sh failed with exit code $EXIT_CODE."; exit $EXIT_CODE; fi
echo "Matching script completed."

# --- Execute Selection Writing ---
echo "Starting selection writing script (write_selected_data.py)..."
python3 -m less.data_selection.write_selected_data \
    --target_task_names ${TARGET_TASK_NAMES} \
    --train_file_names ${TRAIN_FILE_NAMES} \
    --train_files ${OMI1_TRAIN_FILE} \
    --output_path ${SELECTED_DATA_OUTPUT_PATH} \
    --percentage ${SELECTION_PERCENTAGE}

EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then  echo "Error: write_selected_data.py failed with exit code $EXIT_CODE."; exit $EXIT_CODE; fi
echo "Selection writing script completed."

# --- Optional: Rename Output File for Clarity ---
OUTPUT_FILE_PATTERN="${SELECTED_DATA_OUTPUT_PATH}/${TARGET_TASK_NAMES}/top_p*.jsonl"
OUTPUT_FILE_TARGET="${SELECTED_DATA_OUTPUT_PATH}/${TARGET_TASK_NAMES}/top_1000.jsonl"
GENERATED_FILE=$(ls -t ${OUTPUT_FILE_PATTERN} 2>/dev/null | head -n 1) # Avoid error message if no file found

if [ -n "$GENERATED_FILE" ] && [ -f "$GENERATED_FILE" ]; then
   echo "Renaming $GENERATED_FILE to $OUTPUT_FILE_TARGET"
   mv "$GENERATED_FILE" "$OUTPUT_FILE_TARGET"
else
   echo "Warning: Could not find output file matching pattern $OUTPUT_FILE_PATTERN to rename."
fi

echo "Finished Step 3B: Matching and Selection."
echo "Wrapper script finished with status $?. Check Slurm output files."
# --- End of Job ---