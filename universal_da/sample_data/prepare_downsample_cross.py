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
from universal_da.simple_cross.convert_dataset_uniform import new_unified_example,get_emap,update_example_with_emap,update_example_with_additional_constraints,build_wiki_uniformed_dataset
import copy
import random
from universal_da.dataset_config import default_T0_tasks
from multiprocessing import Pool

multi_data_dir='../../huggingface_datasets'
save_name='details_CrossOrigindataset_crossed_correct_balanced_0618'
save_dir='../../data/huggingface_datasets_0425'

def replace_paragraph(examples):
    ret_examples=[]
    for e in examples:
        new_e=e
        if new_e['para1']==None:
            new_e['para1']=new_e['para2']
            new_e['para2']=None
        ret_examples.append(new_e)
    return ret_examples

def get_sub_qa_dataset(examples,only_True=True, multi_choice=False, limit_num=1000):
    # if the task_type is 'qa_tf' we return examples if and only if "False answers" exists
    ret_examples=[] if only_True==True else {'True':[],'False':[]}
    for e in examples:
        new_e=copy.deepcopy(e)
        question_list=new_e.pop('questions')
        answers_list=new_e.pop('answers')
        for question, answers in zip(question_list,answers_list):
            new_e=copy.deepcopy(e)
            _=new_e.pop('questions')
            _=new_e.pop('answers')
            new_e['question']=question
            if new_e['para1'] is None: new_e['para1']=new_e['para2']
            if multi_choice==True and len(answers['wrong'])==0: continue
            if len(answers['correct'])==0: 
                answers['correct'].append('no answer')
            choosen_answer=np.random.choice(answers['correct'])
            choices=[choosen_answer]+np.random.choice(answers['wrong'],min(limit_num-1,len(answers['wrong'])),False).tolist()
            random.shuffle(choices)
            new_e['choices']=choices
            if only_True==True:
                new_e['answer']=choosen_answer
                new_e['answer_label']=1
                ret_examples.append(new_e)
            else:
                if len(answers['correct'])!=0:
                    nnew_e=copy.deepcopy(new_e)
                    nnew_e['answer']=choosen_answer
                    nnew_e['answer_label']=1
                    ret_examples['True'].append(nnew_e)
                if len(answers['wrong'])!=0:
                    nnew_e=copy.deepcopy(new_e)
                    nnew_e['answer']=np.random.choice(answers['wrong'])
                    nnew_e['answer_label']=0
                    ret_examples['False'].append(nnew_e)
    if only_True==False:
        minlen=min(len(ret_examples['True']),len(ret_examples['False']))
        true_examples=np.random.choice(ret_examples['True'],minlen,False)
        false_examples=np.random.choice(ret_examples['False'],minlen,False)
        ret_examples=true_examples.tolist()+false_examples.tolist()
    # print("return examples",ret_examples)
    return ret_examples

task_names=sum([y for (x,y) in default_T0_tasks.items()],[])
myprompts={}
inverse_prompt_map={}
for (x,y) in uniformed_prompt_templates.items():
    if x in ['question_paraphrase_tf']: continue
    myprompts={**myprompts,**y}
    for z in y.keys():
        inverse_prompt_map[z]=x

max_train_num_per_dataset=50000
max_valid_num_per_dataset=10
large_num=500000
small_num=50000

from universal_da.simple_cross.generate_dataset import prepare_prompt_codes, prepare_example_codes
myprompts_name_list=[]
myprompts_list=[]
for (x,y) in myprompts.items():
    myprompts_name_list.append(x);myprompts_list.append(y)
prompt_codes,all_match_value=prepare_prompt_codes(myprompts_list)


downsample_task_names=['cosmos_qa','adversarial_qa/dbidaf','adversarial_qa/droberta','quartz','social_i_qa','kilt_tasks/hotpotqa','adversarial_qa/dbert','ropes','quail']
################################################################################################
# build dataset and save to disk for the sake of time (if you want to run with several repeat times, you can merge the two steps if don't want to save
################################################################################################
# step1

# def build_cross_data(task_name):
#     print(task_name)
#     np.random.seed(42)
#     random.seed(42)
#     tmp_total_train_examples=[]
#     if task_name in ['glue/qqp','wiqa','wiki_hop/original','trec','wiki_bio']:
#         return False
#     else:
#         if os.path.exists(os.path.join('../../spec_datasets',save_name,task_name)):
#             return False
#         train_examples,valid_dataset=T0_data_utils.load_datasets(data_dir=multi_data_dir,task_name=task_name,train_num=-1,valid_num=max_valid_num_per_dataset,k_fold=-1,fold_id=-1)
#         if task_name not in downsample_task_names and len(train_examples)>small_num:
#             prompts = DatasetTemplates(task_name)
#             total_prompt_names=[x for x in prompts.name_to_id_mapping]
#             prompt_num=len(total_prompt_names)
#             rng=random.Random(42)
#             train_idxes=rng.sample(range(len(train_examples)),50000//prompt_num)
#             train_examples=train_examples.select(train_idxes)
#             print('task_to_be_downsampled: {}, {}'.format(task_name,len(train_examples)))
#         print(len(train_examples))
#         if task_name=='wiki_qa':
#             uniformed_dataset=build_wiki_uniformed_dataset(train_examples)
#             uniformed_dataset=[x for x in uniformed_dataset if x['answers'][0]['correct']!=[]]
#         else:
#             uniformed_dataset=[]
#             for orig_e in train_examples:
#                 uni_e=new_unified_example()
#                 emap=get_emap(task_name)
#                 uni_e=update_example_with_emap(uni_e,orig_e,emap)
#                 uni_e=update_example_with_additional_constraints(uni_e,orig_e,task_name)
#                 uniformed_dataset.append(uni_e)
#         uniformed_dataset=replace_paragraph(uniformed_dataset)
#         for (prompt_name,prompt_code) in zip(myprompts_name_list,prompt_codes):
#             # print('checking {}'.format(prompt_name))
#             # check if this prompt is suitable
#             if inverse_prompt_map[prompt_name] in ['question_to_answer_tf','title_question_to_answer_tf','paragraph_question_to_answer_tf']:
#                 check_example=get_sub_qa_dataset([uniformed_dataset[0]],only_True=True)
#             elif inverse_prompt_map[prompt_name] in ['question_to_answer','paragraph_question_to_answer','paragraph_question_title_to_answer','answer_title_to_question','paragraph_to_question','paragraph_answer_to_question','question_answer_to_title','answer_to_title','question_answer_to_paragraph','question_to_choose_answer','paragraph_question_to_choose_answer']:
#                 multi_choice=True if 'choose' in inverse_prompt_map[prompt_name] else False
#                 check_example=get_sub_qa_dataset([uniformed_dataset[0]],only_True=True,multi_choice=multi_choice)
#             else:
#                 check_example=uniformed_dataset
#             if len(check_example)==0: continue
#             else: check_example=check_example[0]
#             [example_code]=prepare_example_codes([check_example])
#             if (int(prompt_code[0],2) & int(prompt_code[1],2))|int(example_code,2) != all_match_value: continue
#             print(task_name,prompt_name,inverse_prompt_map[prompt_name])
#             examples=['None']
#             if inverse_prompt_map[prompt_name] in ['question_to_answer_tf','title_question_to_answer_tf','paragraph_question_to_answer_tf']:
#                 new_uniformed_dataset=get_sub_qa_dataset(uniformed_dataset,only_True=False)
#             elif inverse_prompt_map[prompt_name] in ['question_to_answer','paragraph_question_to_answer','paragraph_question_title_to_answer','answer_title_to_question','paragraph_to_question','paragraph_answer_to_question','question_answer_to_title','answer_to_title','question_answer_to_paragraph','question_to_choose_answer','paragraph_question_to_choose_answer']:
#                 multi_choice=True if 'choose' in inverse_prompt_map[prompt_name] else False
#                 if prompt_name in ['do_not_use','logic_test','heres_a_story','choose_between','testing_students','Multiple Choice (Closed Book)']:
#                     limit_num=2
#                 else: limit_num=100
#                 new_uniformed_dataset=get_sub_qa_dataset(uniformed_dataset,only_True=True,multi_choice=multi_choice,limit_num=limit_num)
#             else:
#                 new_uniformed_dataset=uniformed_dataset
#             examples=apply_prompts(new_uniformed_dataset,[myprompts[prompt_name]])
#             examples=datasets.Dataset.from_dict({'input_tokens':[x['input_tokens'] for x in examples],'label':[x['label'] for x in examples]})
#             if len(examples)==0: continue
#             tmp_total_train_examples.append(examples)
#             print(len(examples))
#             print(examples[0])
#         tmp_train_data=datasets.concatenate_datasets(tmp_total_train_examples)
#         tmp_train_data.save_to_disk(os.path.join('../../spec_datasets',save_name,task_name))
#     return True

# pool=Pool(processes=38)
# results=pool.map(build_cross_data,task_names)
# pool.close()

# import pdb 
# pdb.set_trace()
#step2
total_train_examples=[]
total_valid_examples=[]
for i,task_name in enumerate(task_names):
    print(i,task_name)
    np.random.seed(42)
    random.seed(42)
    tmp_total_train_examples=[]
    if task_name in ['glue/qqp','wiqa','wiki_hop/original','trec','wiki_bio']:
        prompts = DatasetTemplates(task_name)
        total_prompt_names=[x for x in prompts.name_to_id_mapping]
        prompt_num=len(total_prompt_names)
        for prompt_name in total_prompt_names:
            data_dir=os.path.join(multi_data_dir,'P3') 
            train_examples,valid_examples=load_datasets_p3(data_dir=data_dir,task_name=task_name,prompt_name=prompt_name,train_num=-1,valid_num=5000,prompt_num=1,k_fold=-1,fold_id=-1)
            new_train=datasets.Dataset.from_dict({'input_tokens':train_examples['inputs_pretokenized'],'label':train_examples['targets_pretokenized']})
            tmp_total_train_examples.append(new_train)
            if valid_examples is not None:
                new_valid=datasets.Dataset.from_dict({'input_tokens':valid_examples['inputs_pretokenized'],'label':valid_examples['targets_pretokenized']})
                total_valid_examples.append(new_valid)
            # print(task_name,len(tmp_train_data))
    else:
        tmp_train_data=datasets.load_from_disk(os.path.join('../../spec_datasets',save_name,task_name))
        # print(task_name,len(tmp_train_data))
        tmp_total_train_examples.append(tmp_train_data)
    tmp_total_train_data=datasets.concatenate_datasets(tmp_total_train_examples)
    print(task_name,len(tmp_total_train_data))
    # if task_name in downsample_task_names:
    #     idxs=list(range(len(tmp_total_train_data)))
    #     orig_examples=datasets.load_from_disk('{}/{}'.format(multi_data_dir,task_name))
    #     if len(orig_examples)>large_num:
    #         sample_idxs=np.random.choice(idxs,large_num,False).tolist()
    #         tmp_total_train_data=tmp_total_train_data.select(sample_idxs)
    if task_name not in downsample_task_names and task_name in ['glue/qqp','wiqa','wiki_hop/original','trec','wiki_bio']:
        idxs=list(range(len(tmp_total_train_data)))
        if len(idxs)>small_num:
            sample_idxs=np.random.choice(idxs,small_num,False).tolist()
            tmp_total_train_data=tmp_total_train_data.select(sample_idxs)
    print(task_name,len(tmp_total_train_data))
    total_train_examples.append(tmp_total_train_data)


import pdb 
pdb.set_trace()
train_data=datasets.concatenate_datasets(total_train_examples)
train_data = train_data.rename_column("input_tokens", "inputs_pretokenized")
train_data = train_data.rename_column("label", "targets_pretokenized")
idxs=list(range(len(train_data)))
num=4000000
if len(idxs)>num: sample_idxs=np.random.choice(idxs,num,False).tolist()
else: sample_idxs=np.random.choice(idxs,num,True).tolist()
train_data=train_data.select(sample_idxs)
train_data.save_to_disk(os.path.join(save_dir,'support_p2_downsample_0610_{}_{}w/train'.format(small_num,num//10000)))

valid_data=datasets.concatenate_datasets(total_valid_examples)
valid_data = valid_data.rename_column("input_tokens", "inputs_pretokenized")
valid_data = valid_data.rename_column("label", "targets_pretokenized")
valid_data.save_to_disk(os.path.join(save_dir,'support_p2_downsample_0610_{}_{}w/validation'.format(small_num,num//10000)))
import pdb 
pdb.set_trace()

# python -m universal_da.sample_data.prepare_downsample_cross


