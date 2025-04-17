#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

echo "======================================================"
echo " Starting LESS Data Selection Script                  "
echo " Source Data: Dolly                                   "
echo " Target Task: AlpacaEval                              "
echo " Selection: Top 1000                                  "
echo "======================================================"
START_TIME=$(date +%s)

# --- Configuration ---
# Adjust these variables to match your setup

# Checkpoints from warmup training to use (space-separated list)
CKPTS_LIST="105 211 317 420"
# Corresponding average learning rates for these checkpoints (MUST match order and count)
CHECKPOINT_WEIGHTS="1.6877e-05 1.2859e-05 7.7030e-06 2.5616e-06"

# Base directory containing the warmup checkpoint folders (e.g., checkpoint-105)
BASE_MODEL_CHECKPOINT_DIR="../out/llama2-7b-p0.05-lora-seed3"
# Base directory to save computed gradients
GRAD_BASE_DIR="../grads/llama2-7b-p0.05-lora-seed3"
# Base directory to save influence scores and the final selected data file
SELECTED_DATA_OUTPUT_PATH="../selected_data"
# Main data directory (containing train/ and eval/)
DATA_DIR="./data"

# Source Training Data Config
TRAINING_DATA_NAME="dolly"
TRAINING_DATA_FILE="${DATA_DIR}/train/processed/${TRAINING_DATA_NAME}/${TRAINING_DATA_NAME}_data.jsonl"
# Gradient type for training data (usually 'adam')
TRAIN_GRAD_TYPE="adam"

# Target Evaluation Data Config
TARGET_TASK_NAME="alpacaevel"
# Gradient type for eval data (always 'sgd')
EVAL_GRAD_TYPE="sgd"

# LESS Parameters
DIMS_TRAIN="8192"       # Dimensions for training gradients (can be "4096 8192")
DIMS_EVAL="4096 8192" # Dimensions for eval gradients
DIM_MATCHING=8192     # Dimension to use for matching step (must exist in DIMS_TRAIN/DIMS_EVAL)
TOP_K=1000            # Select top K examples <-- SET TO 1000

# --- End Configuration ---

# --- Ensure output directories exist ---
mkdir -p "${GRAD_BASE_DIR}"
mkdir -p "${SELECTED_DATA_OUTPUT_PATH}"

echo "INFO: Using Checkpoints: ${CKPTS_LIST}"
echo "INFO: Output Gradient Directory: ${GRAD_BASE_DIR}"
echo "INFO: Output Selected Data Directory: ${SELECTED_DATA_OUTPUT_PATH}"
echo "-------------------------------------"

# Step 2: Build Gradient Datastore for Dolly (Loop over checkpoints)
echo "STEP 2: Calculating gradients for Training Data (${TRAINING_DATA_NAME})..."
for CKPT in $CKPTS_LIST; do
    MODEL_PATH="${BASE_MODEL_CHECKPOINT_DIR}/checkpoint-${CKPT}"
    OUTPUT_PATH="${GRAD_BASE_DIR}/${TRAINING_DATA_NAME}-ckpt${CKPT}-${TRAIN_GRAD_TYPE}"
    # Basic check if final dimension output already exists to allow resuming
    if [ -d "${OUTPUT_PATH}/dim${DIM_MATCHING}" ]; then
         echo "  Skipping Dolly grads for CKPT ${CKPT} - Output directory seems to exist: ${OUTPUT_PATH}/dim${DIM_MATCHING}"
    else
         echo "  Running Dolly grads for CKPT ${CKPT}..."
         ./less/scripts/get_info/grad/get_train_lora_grads.sh \
            "$TRAINING_DATA_FILE" \
            "$MODEL_PATH" \
            "$OUTPUT_PATH" \
            "$DIMS_TRAIN" \
            "$TRAIN_GRAD_TYPE"
        echo "  Finished Dolly grads for CKPT ${CKPT}."
    fi
done
echo "STEP 2: Done."
echo "-------------------------------------"

# Step 3b: Get Evaluation Gradients for AlpacaEval (Loop over checkpoints)
echo "STEP 3b: Calculating gradients for Target Task (${TARGET_TASK_NAME})..."
for CKPT in $CKPTS_LIST; do
    MODEL_PATH="${BASE_MODEL_CHECKPOINT_DIR}/checkpoint-${CKPT}"
    OUTPUT_PATH="${GRAD_BASE_DIR}/${TARGET_TASK_NAME}-ckpt${CKPT}-${EVAL_GRAD_TYPE}"
    # Basic check if final dimension output already exists
    if [ -d "${OUTPUT_PATH}/dim${DIM_MATCHING}" ]; then
         echo "  Skipping ${TARGET_TASK_NAME} grads for CKPT ${CKPT} - Output directory seems to exist: ${OUTPUT_PATH}/dim${DIM_MATCHING}"
     else
        echo "  Running ${TARGET_TASK_NAME} grads for CKPT ${CKPT}..."
        ./less/scripts/get_info/grad/get_eval_lora_grads.sh \
            "$TARGET_TASK_NAME" \
            "$DATA_DIR" \
            "$MODEL_PATH" \
            "$OUTPUT_PATH" \
            "$DIMS_EVAL"
        echo "  Finished ${TARGET_TASK_NAME} grads for CKPT ${CKPT}."
    fi
done
echo "STEP 3b: Done."
echo "-------------------------------------"

# Step 3c: Select Data (Matching)
echo "STEP 3c: Matching gradients..."
GRADIENT_PATH_TEMPLATE="${GRAD_BASE_DIR}/${TRAINING_DATA_NAME}-ckpt{}-${TRAIN_GRAD_TYPE}/dim${DIM_MATCHING}"
VALIDATION_GRADIENT_PATH_TEMPLATE="${GRAD_BASE_DIR}/${TARGET_TASK_NAME}-ckpt{}-${EVAL_GRAD_TYPE}/dim${DIM_MATCHING}"

./less/scripts/data_selection/matching.sh \
    "$GRADIENT_PATH_TEMPLATE" \
    "$TRAINING_DATA_NAME" \
    "$CKPTS_LIST" \
    "$CHECKPOINT_WEIGHTS" \
    "$VALIDATION_GRADIENT_PATH_TEMPLATE" \
    "$TARGET_TASK_NAME" \
    "$SELECTED_DATA_OUTPUT_PATH"
echo "STEP 3c: Done."
echo "-------------------------------------"

# Step 3d: Write Selected Data (Top K)
echo "STEP 3d: Writing selected top ${TOP_K} data points..."
INFLUENCE_SCORES_PATH="$SELECTED_DATA_OUTPUT_PATH" # Path where matching.sh saves scores
FINAL_OUTPUT_PATH="$SELECTED_DATA_OUTPUT_PATH"    # Path where write_selected_data.py saves the final file

python3 -m less.data_selection.write_selected_data \
    --target_task_names ${TARGET_TASK_NAME} \
    --train_file_names ${TRAINING_DATA_NAME} \
    --train_files ${TRAINING_DATA_FILE} \
    --influence_scores_path ${INFLUENCE_SCORES_PATH} \
    --output_path ${FINAL_OUTPUT_PATH} \
    --top_k ${TOP_K} # <-- Use top_k instead of percentage
echo "STEP 3d: Done."
echo "-------------------------------------"

END_TIME=$(date +%s)
ELAPSED_TIME=$((END_TIME - START_TIME))
# Construct the expected final filename (Note: the python script might slightly alter this)
EXPECTED_OUTPUT_FILE="${FINAL_OUTPUT_PATH}/${TARGET_TASK_NAME}/${TRAINING_DATA_NAME}_top_k${TOP_K}.jsonl"

echo "======================================================"
echo " LESS Data Selection Script Finished Successfully!    "
echo " Selected data saved to a file like:                  "
echo " ${EXPECTED_OUTPUT_FILE}"
echo " (Please verify the exact filename in the directory)  "
echo " Total execution time: ${ELAPSED_TIME} seconds.       "
echo "======================================================"