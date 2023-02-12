#! /bin/bash

script_path=$(realpath $BASH_SOURCE)
script_dir=$(dirname $script_path)

load_dir="../../../huggingface_models/t5-xxl-lm-adapt"
multi_data_dir="../../data/huggingface_datasets_0425"
tokenizer_dir=${load_dir}
# tokenizer_dir="../../../huggingface_models/t5-xxl-lm-adapt"

experiment_name="11B-support_original_downsample_500000_50000_0616-bsz-1024-seq-512-seed-1234-"${task_set}-${config_name}

save_dir="../checkpoints/checkpoints-"${experiment_name}
config_json="config/t5_lm/config_t5_lm_xxl.json"
gpt_options=" \
       --bert-mask-ratio 0.15 \
       --avg-block-length 3 \
       --experiment-name ${experiment_name} \
       --model-parallel-size 1 \
       --t5-model \
       --vocab-size 32128 \
       --num-layers 24 \
       --hidden-size 4096 \
       --num-attention-heads 64 \
       --inner-hidden-size 10240 \
       --hidden-size-per-attention-head 64 \
       --seq-length 512 \
       --relative-attention-num-buckets 32 \
       --layernorm-epsilon 1e-6 \
       --gated-gelu-mlp \
       --no-share-embeddings \
       --init-method-std 1.0 \
       --save ${save_dir} \
       --load ${load_dir} \
       --log-interval 50 \
       --eval-interval 1000 \
       --save-interval 1000 \
       --train-iters 20000 \
       --multi-task-ratio 1.0 \
       --shuffle \
       --loader-scatter 8 \
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