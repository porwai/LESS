DATA_DIR=./data
MODEL_PATH="/scratch/network/pw5115/my_less_project/Llama-2-7b-hf"
PERCENTAGE=0.05 # Removed comment just in case
DATA_SEED=3
TRAIN_SET=openmathinstruct1
# Ensure correct expansion if pasting: the P should be uppercase
JOB_NAME="llama2-7b-p${PERCENTAGE}-lora-seed${DATA_SEED}"

# Verify JOB_NAME looks right (optional)
echo "JOB_NAME is set to: ${JOB_NAME}"

# Run the script
./less/scripts/train/warmup_lora_train.sh "$DATA_DIR" "$MODEL_PATH" "$PERCENTAGE" "$DATA_SEED" "$JOB_NAME" "$TRAIN_SET"