#!/bin/bash

source less/scripts/train/base_training_args.sh

data_dir=$1
model_path=$2
percentage=$3
data_seed=$4
job_name=$5
train_set=$6

output_dir=../out/${job_name}
if [[ ! -d $output_dir ]]; then
    mkdir -p $output_dir
fi

train_file="$data_dir/train/processed/${train_set}/${train_set}_data.jsonl"
if [[ ! -f $train_file ]]; then
  echo "❌  Could not find ${train_file}"
  exit 1
fi

# use fsdp for large models
if [[ $model_path == "meta-llama/Llama-2-13b-hf" ]]; then
    base_training_args="$base_training_args --fsdp 'full_shard auto_wrap' --fsdp_config llama2_13b_finetune"
    elif [[ $model_path == "mistralai/Mistral-7B-v0.1" ]]; then
    base_training_args="$base_training_args --fsdp 'full_shard auto_wrap' --fsdp_config mistral_7b_finetune"
fi


training_args="$base_training_args \
  --model_name_or_path $model_path \
  --output_dir $output_dir \
  --percentage $percentage \
  --data_seed $data_seed \
  --train_files $train_file"

echo "🚀  Launching warm‑up with:"
echo "    Model        : $model_path"
echo "    Train set    : $train_set"
echo "    Data fraction: $percentage"
echo "    Seed         : $data_seed"
echo "    Output dir   : $output_dir"
echo

eval "$header" "$training_args" 2>&1 | tee "$output_dir/train.log"