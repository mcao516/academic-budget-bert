#!/bin/bash
module load StdEnv/2020 gcc/9.3.0 cuda/11.0
module load arrow/5.0.0
module load python/3.8
source $HOME/envABERT/bin/activate

MODEL_NAME_OR_PATH=$SCRATCH/academic-BERT-48h/pretraining_experiment-/epoch1000000_step47703

python run_glue.py \
  --model_name_or_path $MODEL_NAME_OR_PATH \
  --task_name SST2 \
  --max_seq_length 128 \
  --output_dir /tmp/finetuning \
  --overwrite_output_dir \
  --do_train --do_eval \
  --evaluation_strategy steps \
  --per_device_train_batch_size 32 --gradient_accumulation_steps 1 \
  --per_device_eval_batch_size 32 \
  --learning_rate 5e-5 \
  --weight_decay 0.01 \
  --eval_steps 50 --evaluation_strategy steps \
  --max_grad_norm 1.0 \
  --num_train_epochs 5 \
  --lr_scheduler_type polynomial \
  --warmup_steps 50;
