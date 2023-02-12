import numpy as np 
import datasets
import random
import os
data_dir='/share/zongyu/zhoujing/huggingface_datasets/jing_crossed_dataset_unfiltered/aug_T0_v8'
save_dir='/share/zongyu/data/huggingface_datasets_0425'
global_rng=random.Random(42)
################################################################################
# sentiment, domain are determined by the paragraphs (might be extremely unbalanced among datasets), so we balance each category after merging all the datasets.
################################################################################

# task_types=['sentiment2','sentiment5','topic1','topic2']
task_types=['sentiment2_small','sentiment5_small','topic1_small','topic2_small']
for task_type in task_types:
    candidates0=list(np.load(os.path.join(data_dir,task_type,'candidates_0.npy'),allow_pickle=True))
    candidates1=list(np.load(os.path.join(data_dir,task_type,'candidates_1.npy'),allow_pickle=True))
    all_data=candidates0+candidates1
    cand_k = {"inputs_pretokenized":[],"targets_pretokenized":[]}
    cand_k['inputs_pretokenized']=[x['input_tokens'] for x in all_data]
    cand_k['targets_pretokenized']=[x['label'] for x in all_data]

    print(task_type,len(cand_k['inputs_pretokenized']))
    cand_ds=datasets.Dataset.from_dict(cand_k)
    cand_ds.save_to_disk(os.path.join(save_dir,'unida_{}_0601/train'.format(task_type)))



###################################################################################################
# build data for other task types
# limit each dataset with an upper bound 50,000
# small: 2W, large: 10W, tiny: 1W
###################################################################################################

# MAX_SIZE=10000
# def _sample_train_data(task_name, train_split,rng):
#     train_number = len(train_split)
#     if train_number > MAX_SIZE:
#         sample_train_number = int(MAX_SIZE)
#         sample_train_index_list = rng.sample(range(len(train_split)), k=sample_train_number)
#         samples = train_split.select(sample_train_index_list)
#         return samples
#     else:
#         return train_split

# abandon_task_names={
#     'paraphrase':['glue/mrpc','glue/qqp','paws/labeled_final'],
#     'qa_tf':['cos_e/v1.11','cosmos_qa','dream','qasc','quail','quarel','quartz','sciq','social_i_qa','wiki_hop/original','wiqa','wiki_bio'],
#     'qa_extractive':['adversarial_qa/dbidaf','adversarial_qa/dbert','adversarial_qa/droberta','duorc/SelfRC','duorc/ParaphraseRC','ropes','quoref']+['cos_e/v1.11','cosmos_qa','dream','qasc','quail','quarel','quartz','sciq','social_i_qa','wiki_hop/original','wiqa','wiki_bio'],
#     'qa_multiple_choice':['cos_e/v1.11','cosmos_qa','dream','qasc','quail','quarel','quartz','sciq','social_i_qa','wiki_hop/original','wiqa','wiki_bio'],
#     'keywords':['common_gen','wiki_bio'],
#     'summary':['cnn_dailymail/3.0.0','gigaword','multi_news','samsum','xsum'],
#     'title':[],
# }
# from universal_da.dataset_config import default_T0_tasks
# task_names=[]
# for task_type,names in default_T0_tasks.items():
#     task_names+=names

# task_types=['paraphrase','qa_tf','qa_extractive','qa_multiple_choice','summary','title','keywords']

# total_type_datas={x:[] for x in task_types}
# for task_name in task_names:
#     if task_name in ['glue/qqp','kilt_tasks/hotpotqa','wiki_qa','cos_e/v1.11','quarel','sciq','trec']: continue
#     candidates0=np.load(os.path.join(data_dir,task_name,'candidates_0.npy'),allow_pickle=True).item()
#     candidates1=np.load(os.path.join(data_dir,task_name,'candidates_1.npy'),allow_pickle=True).item()
#     for task_type in task_types:
#         if task_name in abandon_task_names[task_type]: continue
#         all_data=candidates0[task_type]+candidates1[task_type]
#         cand_k = {"inputs_pretokenized":[],"targets_pretokenized":[]}
#         cand_k['inputs_pretokenized']=[x['input_tokens'] for x in all_data]
#         cand_k['targets_pretokenized']=[x['label'] for x in all_data]
#         cand_ds=datasets.Dataset.from_dict(cand_k)
#         cand_ds_ = _sample_train_data(task_name,cand_ds,global_rng)
#         print(task_name,task_type,len(cand_ds),len(cand_ds_))
#         total_type_datas[task_type].append(cand_ds_)

# for task_type in task_types:
#     save_data=datasets.concatenate_datasets(total_type_datas[task_type])
#     print(task_type,len(save_data))
#     save_data.save_to_disk(os.path.join(save_dir,'unida_{}_tiny_0601/train'.format(task_type)))

# python -m universal_da.simple_cross.build_universal_data