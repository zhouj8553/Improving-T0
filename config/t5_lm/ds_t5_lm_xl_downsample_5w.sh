#! /bin/bash

script_path=$(realpath $BASH_SOURCE)
script_dir=$(dirname $script_path)

max_seq_len=512
max_training_steps=20000
total_decay_iters=16000

experiment_name="3B-support_original_downsample_500000_50000_0616-"${task_set}-${max_seq_len}-${config_name}

save_dir="../checkpoints/checkpoints-"${experiment_name}

tok_load_dir="../../../huggingface_models/t5-xl-lm-adapt"
multi_data_dir="../../data/huggingface_datasets_0425"
config_json="../../config_t5_lm_xl_4.json"
gpt_options=" \
       --bert-mask-ratio 0.15 \
       --avg-block-length 3 \
       --experiment-name ${experiment_name} \
       --model-parallel-size 1 \
       --t5-model \
       --vocab-size 32128 \
       --num-layers 24 \
       --hidden-size 2048 \
       --inner-hidden-size 5120 \
       --num-attention-heads 32 \
       --hidden-size-per-attention-head 64 \
       --relative-attention-num-buckets 32 \
       --no-share-embeddings \
       --gated-gelu-mlp \
       --layernorm-epsilon 1e-6 \
       --init-method-std 1.0 \
       --seq-length ${max_seq_len} \
       --shuffle \
       --loader-scatter 8 \
       --save ${save_dir} \
       --load ${tok_load_dir} \
       --log-interval 100 \
       --eval-interval 1000 \
       --save-interval 1000 \
       --train-iters ${max_training_steps} \
       --multi-task-ratio 1.0 \
       --multi-src-seq-length ${max_seq_len} \
       --multi-tgt-seq-length 256 \
       --multi-cache-dir ${multi_data_dir} \
       --multi-task-set ${task_set} \
       --train-data bert-large \
       --tokenizer-type hf_T5Tokenizer \
       --tokenizer-model-type ${tok_load_dir} \
       --distributed-backend nccl \
       --lr-decay-style cosine \
       --lr-decay-iters ${total_decay_iters} \
       --lr-decay-ratio 0.1 \
       --warmup 0.04 \
       --checkpoint-activations \
       --no-deepspeed-load \
       --no-load-optim \
       --no-load-lr-scheduler \
       --no-load-rng \
       --no-load-iteration \
       --fp16 \
"
gpt_options="${gpt_options}
               --deepspeed \
               --deepspeed_config ${config_json} \
"

task_options="\
--t0-prepared-task-names support_original_downsample_500000_50000_0616 \
"

total_options="${gpt_options} \
    ${task_options}
"