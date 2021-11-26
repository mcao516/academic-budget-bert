#!/bin/bash
module load StdEnv/2020 gcc/9.3.0 cuda/11.0
module load arrow/5.0.0
module load python/3.8
source ~/envABERT/bin/activate

DATA_PATH=$SCRATCH/BERT-pretrain-corpus/samples
OUTPUT_PATH=$SCRATCH/academic-BERT-test/
MODEL_NAME_OR_PATH=$SCRATCH/huggingface/bert-large-uncased
mkdir $OUTPUT_PATH

deepspeed run_pretraining.py \
    --model_type bert-mlm --tokenizer_name bert-large-uncased \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --hidden_act gelu \
    --hidden_size 1024 \
    --num_hidden_layers 24 \
    --num_attention_heads 16 \
    --intermediate_size 4096 \
    --hidden_dropout_prob 0.1 \
    --attention_probs_dropout_prob 0.1 \
    --encoder_ln_mode pre-ln \
    --lr 1e-3 \
    --train_batch_size 4096 \
    --train_micro_batch_size_per_gpu 32 \
    --lr_schedule time \
    --curve linear \
    --warmup_proportion 0.06 \
    --gradient_clipping 0.0 \
    --optimizer_type adamw \
    --weight_decay 0.01 \
    --adam_beta1 0.9 \
    --adam_beta2 0.98 \
    --adam_eps 1e-6 \
    --total_training_time 24.0 \
    --early_exit_time_marker 24.0 \
    --dataset_path $DATA_PATH \
    --output_dir $OUTPUT_PATH \
    --print_steps 100 \
    --num_epochs_between_checkpoints 10000 \
    --job_name pretraining_experiment \
    --project_name budget-bert-pretraining \
    --validation_epochs 3 \
    --validation_epochs_begin 1 \
    --validation_epochs_end 1 \
    --validation_begin_proportion 0.05 \
    --validation_end_proportion 0.01 \
    --validation_micro_batch 16 \
    --deepspeed \
    --data_loader_type dist \
    --do_validation \
    --use_early_stopping \
    --early_stop_time 180 \
    --early_stop_eval_loss 6 \
    --seed 42 \
    --fp16;
