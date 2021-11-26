#!/bin/bash

# Data Processing
PATH_TO_XML=$SCRATCH/mcao610/BERT-pretrain-corpus/enwiki-latest-pages-articles.xml
PROCESS_OUTPUT=$SCRATCH/mcao610/BERT-pretrain-corpus/processed
python process_data.py -f $PATH_TO_XML -o $PROCESS_OUTPUT --type wiki;

# Initial Sharding
SHARD_OUTPUT=$SCRATCH/BERT-pretrain-corpus/sharded
python shard_data.py \
    --dir $PROCESS_OUTPUT \
    -o $SHARD_OUTPUT \
    --num_train_shards 256 \
    --num_test_shards 128 \
    --frac_test 0.1;

# Samples Generation
SAMPLE_OUTPUT=$SCRATCH/BERT-pretrain-corpus/samples
python generate_samples.py \
    --dir $SHARD_OUTPUT \
    -o $SAMPLE_OUTPUT \
    --dup_factor 10 \
    --seed 42 \
    --vocab_file None \
    --do_lower_case 1 \
    --masked_lm_prob 0.15 \ 
    --max_seq_length 128 \
    --model_name bert-large-uncased \
    --max_predictions_per_seq 20 \
    --n_processes 16;
