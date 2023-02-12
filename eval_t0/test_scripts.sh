################################################################################################################
# first, check the model name or path
model_name_or_path="../../huggingface_models/t5-xl-lm-adapt"
ckpt_path="../T0_related_ckpts/checkpoints-3B-baseline/t5-large-lm-adapt/1000"
eval_data_dir="../../huggingface_datasets"
################################################################################################################
# If you want to test the performance of the zero-shot test datasets of T0, run the following command
python -m torch.distributed.launch --nproc_per_node=8 eval_t0/eval_main.py \
--model_name_or_path ${model_name_or_path} \
--eval_model_dir ${ckpt_path} \
--eval_data_dir ${eval_data_dir} \
--per_device_eval_batch_size 4 \
--output_dir 'results'

# or 
accelerate launch eval_t0/eval_main.py \
--model_name_or_path ${model_name_or_path} \
--eval_model_dir '/share/zongyu/zhoujing/t0_1229_condensed/checkpoints_new/checkpoints-3B-part15-part2zy2-part2zj2-paraphrase5-bsz-1024-seq-512-seed-1234-t0-/t5-xl-lm-adapt/5000' \
--eval_data_dir ${eval_data_dir} \
--per_device_eval_batch_size 4 \
--output_dir 'results'

################################################################################################################
# If you want to test the performance of all training datasets of T0, run the following command
accelerate launch eval_t0/eval_main_train.py \
--model_name_or_path ${model_name_or_path} \
--eval_model_dir '/share/zongyu/zhoujing/spec_checkpoints/mt-t5-lm-large-T0_adam_0.0001_default-T0-tasks_trisoall_validisorand_1_mrpc_add/t5-large-lm-adapt/1000' \
--eval_data_dir ${eval_data_dir} \
--per_device_eval_batch_size 4 \
--output_dir 'test_results'


