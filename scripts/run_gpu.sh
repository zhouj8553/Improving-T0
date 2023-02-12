# single node
bash scripts/ds_pretrain_nvidia_t5.sh config/t5_lm/ds_t5_lm_large_baseline.sh t0
bash scripts/ds_pretrain_nvidia_t5.sh config/t5_lm/ds_t5_lm_xl_baseline.sh t0
bash scripts/ds_pretrain_nvidia_t5.sh config/t5_lm/ds_t5_lm_xxl_baseline.sh t0

bash scripts/ds_pretrain_nvidia_t5.sh config/t5_lm/ds_t5_lm_large_only_top8.sh t0

# multiple nodes
bash scripts/ds_pretrain_nvidia_t5_multi_gpu.sh config/t5_lm/ds_t5_lm_large_baseline.sh t0 2 8
bash scripts/ds_pretrain_nvidia_t5_multi_gpu.sh config/t5_lm/ds_t5_lm_xl_baseline.sh t0 4 8
bash scripts/ds_pretrain_nvidia_t5_multi_gpu.sh config/t5_lm/ds_t5_lm_xxl_baseline.sh t0 8 8

# bash scripts/run_gpu.sh