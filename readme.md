# Improving-T0

This repository contains the official code for the paper "NOT ALL TASKS ARE BORN EQUAL: UNDERSTANDING ZERO-SHOT GENERALIZATION".

This repo implements the process of prompted multi-task training and evaluation.

It is still unclear why our reproduced results are better than that of T0. A small difference between them and us might be that we do not use the packing strategy, and an \<EOS\> is added at the end of the input sentence as is normally used in T5. 

## 0. Data Format Description

We first state the datasets and data format. We use exactly the same datasets as T0. All data in our experiments are in the format of hugging face ```datasets```. In detail, all datasets are saved locally using ```datasets.save_to_disk()```, and can be loaded using ```datasets.load_from_disk()```.

## 1. Train T0

We provide two ways for training the model. One is to load all datasets using a predefined data loader, which is storage-friendly but a little slow to load all datasets one by one. And another is to prepare and save the datasets first, which is easy to implement and faster for loading but requires more disk space.



<!-- First, convert the model from hugging face format into our format using ```python move_t5.py <model_path> <model_name>```. -->

Examples of commands for training are in the file ```scripts/run_gpu.sh```.

We support multi-GPU training on several nodes/machines. Let <run_script> denote the script which controls the number of GPUs we use, and <task_spec_script> denotes the script which control the task-corrected details, such as the model, the training data, and so on. 

1. If you want to train on one machine, execute ```bash <run_script> <task_spec_script> t0``` in the command line. 
2. If you want to use multiple machines for training, please execute ```bash <run_script> <task_spec_script> t0 <num_workers> <num_gpus_per_worker>``` in the command line, where <num_workers> denotes the number of nodes/machines, and <num_gpus_per_worker> denotes how many GPUs you use. Also, please change the IPs accordingly in the file ```hostfiles/hostfile```.

Note that the GPU memory of different machines varies a lot. Adjusting the hyper-parameters according to the GPU memory is essential. The total batch size equals <num_workers> * <num_gpus_per_worker> * accumulate steps * batch size per GPU. The last two hyper-parameters are defined in the folder ```config```. The performance on different machines could be slightly different, but should not vary a lot.

#### 1.1 Train the model with a predefined data loader.

Our predefined data loader is T0, as in scripts that end with _baseline_. You could rewrite the code for more combinations in ```config_data.py```.

For example, a command line to reproduce the baseline is 
```
bash scripts/ds_pretrain_nvidia_t5.sh config/t5_lm/ds_t5_lm_large_baseline.sh t0
```

#### 1.2 Train the model with prepared data.

##### Step1: prepare the data

A set of files for preparing the data can be found in the folder ```universal_da/sample_data```. For example, you could run the following command to generate the dataset with only the top 8 dominant tasks.

```
python -m universal_da.sample_data.prepare_only
```

##### Step2: train the model with prepared data

When we have prepared the data in the format of hugging face datasets, we could use them in our training process and adjust the sampling ratio by adding the following instructions into the script. Let <data_name>s denote a series of the prepared data name. ```t0-prepared-task-names``` controls the task names we need to control, ```t0-upsample-task-names``` controls which tasks should be upsampled, and ```t0-upsample-times``` controls how many times each task should upsample.

```
--t0-prepared-task-names <data_name[1]> <data_name[2]> ...<data_name[n]>
--t0-upsample-task-names <data_name[s1]> <data_name[s2]>
--t0-upsample-times <sample_times[s1]> <sample_times[s2]>
```

If the prepared data has no "validation set", or you don't want to cut the number of the dataset, you should register the dataset name in ```special_tasks_config.py``` before training.

A command line to reproduce the top 8 results is:
```
bash scripts/ds_pretrain_nvidia_t5.sh config/t5_lm/ds_t5_lm_large_only_top8.sh t0
```

## 2. Evaluate Zero-Shot Performance

We implement the evaluation part based on the code released at https://github.com/bigscience-workshop/t-zero/tree/master/evaluation. 

We (1) fix a small bug of T0 (In the original code, when evaluating using multiple GPUs, a small part of the data will duplicate in the last batch to keep the same shape.), and then (2) support the tasks with a variable number of labels (such as wikihop/original), and also support more evaluation metrics for generation tasks (for example, provide _Exact Match_ and _F1_ for extractive QA tasks.).

Denote the model name <model_name_or_path> (The original model path in hugging face.co, where we will load the config and tokenizer. For example, it could be "t5-large-lm-adapt".), and the checkpoint to be evaluated \<ckpt>. Run the following commands for evaluation.

For ease of use, refer to the "eval_t0/test_scripts.sh".

1. Evaluate those datasets in the original test datasets.

```
accelerate launch eval_t0/eval_main.py \
--model_name_or_path <model_name_or_path> \
--eval_model_dir <ckpt> \
--eval_data_dir '../huggingface_datasets' \
--per_device_eval_batch_size 4 \
--output_dir 'test_results'
```

1. Evaluate those datasets in the original training datasets. (In our experiments, it is used in the train-train pairwise evaluation.)

```
for TASK_NAME in $TASK_NAMES
do
	accelerate launch eval_t0/eval_main_train.py \
	--model_name_or_path <model_name_or_path> \
	--eval_model_dir <ckpt> \
	--eval_data_dir '../huggingface_datasets' \
	--per_device_eval_batch_size 1 \
	--output_dir 'single_results'
wait
done
```

## Citation

Please cite us if it is useful in your work:

@inproceedings{zhounot,
  title={Not All Tasks Are Born Equal: Understanding Zero-Shot Generalization},
  author={Zhou, Jing and Lin, Zongyu and Zheng, Yanan and Li, Jian and Yang, Zhilin},
  booktitle={International Conference on Learning Representations}
}
