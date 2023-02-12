import os
from random import uniform
import datasets
import pdb 
import numpy as np
import torch
from universal_da.T0_data_utils import load_datasets_p3
from promptsource.templates import DatasetTemplates
from universal_da.simple_cross.uniform_prompts import uniformed_prompt_templates
from universal_da.simple_cross.choose_candidate import apply_prompts
from universal_da import T0_data_utils
from universal_da.simple_cross.convert_dataset_uniform import new_unified_example,get_emap,update_example_with_emap,update_example_with_additional_constraints
import copy
import random
from universal_da.dataset_config import default_T0_tasks

multi_data_dir='../../huggingface_datasets'
save_name='details_CrossOrigindataset_crossed_correct_balanced_0523'
save_dir='../../data/huggingface_datasets_0425'

task_names=sum([y for (x,y) in default_T0_tasks.items()],[])

max_train_num_per_dataset=50000
max_valid_num_per_dataset=10

repeat_times=2
repeat_task_names=['cosmos_qa','adversarial_qa/dbidaf','adversarial_qa/droberta','quartz','social_i_qa','kilt_tasks/hotpotqa','adversarial_qa/dbert','ropes','quail']
################################################################################################
# build dataset and save to disk for the sake of time (if you want to run with several repeat times, you can merge the two steps if don't want to save
################################################################################################
total_train_examples=[]
total_valid_examples=[]
for i,task_name in enumerate(task_names):
    print(i,task_name)
    np.random.seed(42)
    random.seed(42)
    tmp_total_train_examples=[]
    prompts = DatasetTemplates(task_name)
    total_prompt_names=[x for x in prompts.name_to_id_mapping]
    prompt_num=len(total_prompt_names)
    for prompt_name in total_prompt_names:
        data_dir=os.path.join(multi_data_dir,'P3') 
        train_examples,valid_examples=load_datasets_p3(data_dir=data_dir,task_name=task_name,prompt_name=prompt_name,train_num=50000,valid_num=5000,prompt_num=prompt_num,k_fold=-1,fold_id=-1)
        new_train=datasets.Dataset.from_dict({'inputs_pretokenized':train_examples['inputs_pretokenized'],'targets_pretokenized':train_examples['targets_pretokenized']})
        if task_name in repeat_task_names:
            for _ in range(repeat_times):
                print('repeat {}'.format(task_name))
                total_train_examples.append(new_train)
        else:
            total_train_examples.append(new_train)
        if valid_examples is not None:
            new_valid=datasets.Dataset.from_dict({'inputs_pretokenized':valid_examples['inputs_pretokenized'],'targets_pretokenized':valid_examples['targets_pretokenized']})
            total_valid_examples.append(new_valid)

train_data=datasets.concatenate_datasets(total_train_examples)
# train_data = train_data.rename_column("input_tokens", "inputs_pretokenized")
# train_data = train_data.rename_column("label", "targets_pretokenized")
# idxs=list(range(len(train_data)))
# num=8000000
# if len(idxs)>num: sample_idxs=np.random.choice(idxs,num,False).tolist()
# else: sample_idxs=np.random.choice(idxs,num,True).tolist()
# train_data=train_data.select(sample_idxs)
train_data.save_to_disk(os.path.join(save_dir,'support_original_upsample_{}_0601/train'.format(repeat_times)))

valid_data=datasets.concatenate_datasets(total_valid_examples)
# valid_data = valid_data.rename_column("input_tokens", "inputs_pretokenized")
# valid_data = valid_data.rename_column("label", "targets_pretokenized")
valid_data.save_to_disk(os.path.join(save_dir,'support_original_upsample_{}_0601/validation'.format(repeat_times)))
import pdb 
pdb.set_trace()

# python -m universal_da.sample_data.prepare_upsample_original


