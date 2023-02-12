#! /bin/bash

script_path=$(realpath $BASH_SOURCE)
script_dir=$(dirname $script_path)

load_dir="../../../huggingface_models/t5-large-lm-adapt"
multi_data_dir="../../huggingface_datasets/P3"
tokenizer_dir=${load_dir}
# tokenizer_dir="../../../huggingface_models/t5-large-lm-adapt"

experiment_name="large-baseline-bsz-1024-seq-512-seed-1234-"${task_set}-${config_name}

save_dir="../checkpoints/checkpoints-"${experiment_name}
config_json="config/t5_lm/config_t5_lm_large.json"
gpt_options=" \
       --bert-mask-ratio 0.15 \
       --avg-block-length 3 \
       --experiment-name ${experiment_name} \
       --model-parallel-size 1 \
       --t5-model \
       --vocab-size 32128 \
       --num-layers 24 \
       --hidden-size 1024 \
       --inner-hidden-size 2816 \
       --num-attention-heads 16 \
       --hidden-size-per-attention-head 64 \
       --relative-attention-num-buckets 32 \
       --no-share-embeddings \
       --gated-gelu-mlp \
       --layernorm-epsilon 1e-6 \
       --init-method-std 1.0 \
       --seq-length 512 \
       --shuffle \
       --loader-scatter 8 \
       --save ${save_dir} \
       --load ${load_dir} \
       --log-interval 100 \
       --eval-interval 1000 \
       --save-interval 1000 \
       --train-iters 20000 \
       --multi-task-ratio 1.0 \
       --multi-src-seq-length 512 \
       --multi-tgt-seq-length 256 \
       --multi-cache-dir ${multi_data_dir} \
       --multi-task-set ${task_set} \
       --train-data bert-large \
       --tokenizer-type hf_T5Tokenizer \
       --tokenizer-model-type ${tokenizer_dir} \
       --distributed-backend nccl \
       --lr-decay-style cosine \
       --lr-decay-iters 16000 \
       --lr-decay-ratio 0.1 \
       --warmup 0.04 \
       --checkpoint-activations \
       --no-deepspeed-load \
       --no-load-optim \
       --no-load-lr-scheduler \
       --no-load-rng \
       --no-load-iteration \
       --fp16 \
       --seed 1234 \
"
gpt_options="${gpt_options}
               --deepspeed \
               --deepspeed_config ${config_json} \
"


total_options="${gpt_options} \
    ${task_options}
"