#! /bin/bash
#SBATCH --gres=gpu:1

echo eval_g
source activate py36
# UPDATE PATHS BEFORE RUNNING SCRIPT
export CODE_PATH=/
# absolute path to modeling directory
export DATA_PATH=/
# absolute path to data directory
export CKPT_PATH=/

export TASK_NAME=factcc_annotated
export MODEL_NAME=bert-base-uncased
# Evaluate FactCC model

python3 $CODE_PATH/run.py \
  --do_eval \
  --eval_all_checkpoints \
  --do_lower_case \
  --overwrite_cache \
  --max_seq_length 512 \
  --per_gpu_train_batch_size 12 \
  --model_type bertf \
  --model_name_or_path $MODEL_NAME \
  --data_dir $DATA_PATH \
  --output_dir $CKPT_PATH 
