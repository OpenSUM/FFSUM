#! /bin/bash
#SBATCH --gres=gpu:1
# Train FactCC model

source activate py36
export CODE_PATH=/home/LAB/zhuhd/projects/factcc/modeling2
export DATA_PATH=/home/LAB/zhuhd/projects/factcc/data_input/cnn_data
export OUTPUT_PATH=/home/LAB/zhuhd/projects/factcc/output2/

export MODEL_NAME=bert-base-uncased

python3 -u $CODE_PATH/run.py \
  --do_train \
  --do_eval \
  --do_lower_case \
  --train_from_scratch \
  --data_dir $DATA_PATH \
  --model_type bertf \
  --model_name_or_path $MODEL_NAME \
  --max_seq_length 512 \
  --per_gpu_train_batch_size 12 \
  --learning_rate 8e-6 \
  --num_train_epochs 20.0 \
  --evaluate_during_training \
  --eval_all_checkpoints \
  --overwrite_cache \
  --tokenizer_name bert-base-uncased \
  --output_dir $OUTPUT_PATH/$MODEL_NAME-train-$RANDOM/ \
  --loss_alpha 0.5
    
