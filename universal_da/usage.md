# Taxonomy Tree Based Data Augmentation
# Universal Data Augmentation
## Generate examples with T5
We first sample the data into two parts, and train a model with the data for each part, and then label the other.
To reduce your workload, we released this data at xxx, if you don't want to generate data yourself (It costs a lot of time).

## Choose candidates for two folds seperately
data will be saved at "jing_crossed_dataset_unfiltered/aug_T0_v8". 

`
python -m universal_da.simple_cross.choose_candidate \
--load ../../huggingface_models/t5-large-lm-adapt \
--multi-cache-dir ../../huggingface_datasets \
--eval-batch-size 128 \
--multi-src-seq-length 1024 \
--multi-tgt-seq-length 256 \
--test-ckpt /share/zongyu/zhoujing/T0-Multi-Task_zj-adapt/checkpoints_genmodel/mt-t5-lm-large-T0_adam_0.0001_default-T0-tasks_trisoall_validisorand_1/1500 \
--deepspeed \
--deepspeed_config "config/t5_lm/config_t5_lm_large.json" \
--save test \
--no-deepspeed-load \
--no-load-optim \
--no-load-lr-scheduler \
--no-load-rng \
--no-load-iteration \
--bert-mask-ratio 0.15 \
--avg-block-length 3 \
--experiment-name test \
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
--tokenizer-type hf_T5Tokenizer \
--tokenizer-model-type "/share/zongyu/huggingface_models/t5-large-lm-adapt" \
--k_fold 2 \
--fold_id 1
`
## further sample the data of different parts.
In this part, we only use generated data.
will be saved as datasets in /share/zongyu/data/huggingface_datasets_0425

`
python -m unifersal_da.simple_cross.build_universal_data
`