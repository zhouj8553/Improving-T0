import os
import re
import string 
import random
import logging
import numpy as np

import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from promptsource.templates import DatasetTemplates
from datasets import load_from_disk

from utils import print_rank_0
logger = logging.getLogger(__name__)


def load_datasets(data_dir='../huggingface_datasets',task_name='glue/mrpc',train_num=50000,valid_num=5000,keep_origin=False,seed=42,k_fold=-1,fold_id=-1):
    dataset=load_from_disk(os.path.join(data_dir,task_name))
    print(task_name)
    rng=random.Random(seed)
    if task_name in ['app_reviews','amazon_polarity', 'imdb','yelp_review_full','ag_news','dbpedia_14','trec']: # they have no validation set
        if k_fold!=-1:
            train_examples=dataset['train']
            valid_examples=None
            cached_path=os.path.join(data_dir,'P3','cached_shuffled_idxs','{}_fold{}of{}.npy'.format(task_name,k_fold,fold_id))
            if os.path.exists(cached_path):
                shuffled_idxs=np.load(cached_path)
                print(cached_path,'load_cached')
            else:
                index_path=os.path.join(data_dir,'P3','shuffled_idxs','{}.npy'.format(task_name))
                # print('start load shuffled idxes')
                if os.path.exists(index_path):
                    shuffled_idxs=np.load(index_path)
                else:
                    shuffled_idxs=np.arange(len(train_examples))
                    random.seed(42)
                    random.shuffle(shuffled_idxs)
                    if os.path.exists('/'.join(index_path.split('/')[:-1]))==False:
                        os.makedirs('/'.join(index_path.split('/')[:-1]))
                    np.save(index_path,shuffled_idxs)
                # print('finished load shuffled idxes')
                each_fold_num=len(shuffled_idxs)//k_fold
                shuffled_idxs=shuffled_idxs[each_fold_num*fold_id:each_fold_num*(fold_id+1)]
                shuffled_idxs.sort()
                if os.path.exists('/'.join(cached_path.split('/')[:-1]))==False:
                    os.makedirs('/'.join(cached_path.split('/')[:-1]))
                np.save(cached_path,shuffled_idxs)

            train_examples=train_examples.select(shuffled_idxs)
            if len(train_examples)>train_num and train_num!=-1:
                train_idxes=rng.sample(range(len(train_examples)),train_num)
                train_examples=train_examples.select(train_idxes)
        else:
            if len(dataset['train'])>train_num+valid_num and keep_origin==False and train_num !=-1 and valid_num!=-1:
                total_idxes=rng.sample(range(len(dataset['train'])),train_num+valid_num)
                train_examples=dataset['train'].select(total_idxes[:train_num])
                valid_examples=dataset['train'].select(total_idxes[train_num:])
            else:
                splits=dataset['train'].train_test_split(0.2,seed=seed)
                train_examples=splits['train']
                valid_examples=splits['test']
    elif task_name.split('/')[0]=='story_cloze':
        train_examples=None
        valid_examples=dataset['validation']
        if keep_origin==True:
            return None,valid_examples
        if len(valid_examples)>valid_num and valid_num!=-1:
            valid_idxes=rng.sample(range(len(valid_examples)),valid_num)
            valid_examples=valid_examples.select(valid_idxes)
    else:
        train_examples=dataset['train']
        valid_examples=dataset['val'] if task_name=='wiki_bio' else dataset['validation']
        if k_fold!=-1:
            cached_path=os.path.join(data_dir,'P3','cached_shuffled_idxs','{}_fold{}of{}.npy'.format(task_name,k_fold,fold_id))
            if os.path.exists(cached_path):
                shuffled_idxs=np.load(cached_path)
                print(cached_path,'load_cached')
            else:
                index_path=os.path.join(data_dir,'P3','shuffled_idxs','{}.npy'.format(task_name))
                # print('start load shuffled idxes')
                if os.path.exists(index_path):
                    shuffled_idxs=np.load(index_path)
                else:
                    shuffled_idxs=np.arange(len(train_examples))
                    random.seed(42)
                    random.shuffle(shuffled_idxs)
                    if os.path.exists('/'.join(index_path.split('/')[:-1]))==False:
                        os.makedirs('/'.join(index_path.split('/')[:-1]))
                    np.save(index_path,shuffled_idxs)
                # print('finished load shuffled idxes')
                each_fold_num=len(shuffled_idxs)//k_fold
                shuffled_idxs=shuffled_idxs[each_fold_num*fold_id:each_fold_num*(fold_id+1)]
                shuffled_idxs.sort()
                if os.path.exists('/'.join(cached_path.split('/')[:-1]))==False:
                    os.makedirs('/'.join(cached_path.split('/')[:-1]))
                np.save(cached_path,shuffled_idxs)
            train_examples=train_examples.select(shuffled_idxs)
        if keep_origin==True:
            return train_examples,valid_examples

        # if the number of examples is larger than the required number, just cut it.
        if len(train_examples)>train_num and train_num!=-1:
            train_idxes=rng.sample(range(len(train_examples)),train_num)
            train_examples=train_examples.select(train_idxes)

        if len(valid_examples)>valid_num and valid_num!=-1:
            valid_idxes=rng.sample(range(len(valid_examples)),valid_num)
            valid_examples=valid_examples.select(valid_idxes)
    train_examples=preprocess_dataset(train_examples,task_name) if train_examples is not None else None
    valid_examples=preprocess_dataset(valid_examples,task_name)
    return train_examples, valid_examples

def preprocess_dataset(examples,task_name):
    if examples is None: return examples
    def del_reserved_words(e): # delete reserved words such as "|||" or the prompts might be wrong
        new_heads=[];new_values=[]
        for (header_name,header_value) in zip(e['input_text']['table']['column_header'],e['input_text']['table']['content']):
            if '|||' not in header_name:
                new_heads.append(header_name);new_values.append(header_value)
        if len(e['input_text']['table']['column_header'])!=len(new_heads): 
            e['input_text']['table']['row_number']=[1 for _ in range(len(new_heads))]
            e['input_text']['table']['column_header']=new_heads
            e['input_text']['table']['content']=new_values
        return e

    def replace_with_underline(e): # replace '-' with '_', or the prompt can not parse it.
        new_e={'label_coarse':e['label-coarse'],'label_fine':e['label-fine'],'text':e['text']}
        return new_e

    if task_name=='wiki_bio':
        examples=examples.map(del_reserved_words)
    elif task_name=='trec':
        examples=examples.map(replace_with_underline)
    elif task_name=='samsum':
        examples=examples.filter(lambda example: (len(example['dialogue'])!=0) and (len(example['summary'])!=0))
    # print(len(examples))
    return examples

def preprocess_dataset_p3(examples):
    if examples is None: return None
    def do_strip(e):
        e['inputs_pretokenized']=e['inputs_pretokenized'].strip()
        e['targets_pretokenized']=e['targets_pretokenized'].strip()
        return e
    examples=examples.map(do_strip)
    return examples

def clean(s): # convert the prompt name into the names in P3
    new_s=''
    for c in s:
        if c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789':
            if len(new_s)==0 or new_s[-1]!='_': new_s+='_'
        else: new_s+=c
    return new_s

def load_datasets_p3(data_dir='../huggingface_datasets/P3',task_name='glue/mrpc',prompt_name='',train_num=50000,valid_num=5000,keep_origin=False,seed=42,prompt_num=1,do_eval=False,k_fold=-1,fold_id=-1):
    if task_name.startswith('anli'):
        task_prompt_name='_'.join([task_name.split('/')[0]]+[clean(prompt_name)]+[task_name.split('/')[1]])
    else:
        task_prompt_name='_'.join(task_name.split('/')+[clean(prompt_name)])
    data_path=os.path.join(data_dir,task_prompt_name)
    if os.path.exists(data_path)==False:
        logger.info('Error loading!')
        if do_eval==False:
            raise NotImplementedError
        else:
            return None,None
    if do_eval==True and os.path.exists(data_path+'_score_eval')==False:
        logger.info('Not for evaluate!')
        return None,None
    # print('load_from_disk',data_path)
    dataset=load_from_disk(data_path)
    # print('finished load from disk')
    rng=random.Random(seed)
    if task_name in ['app_reviews']: # they have no validation set
        train_examples=dataset['train']
        valid_examples=None
    elif task_name in ['amazon_polarity', 'imdb','yelp_review_full','ag_news','dbpedia_14','trec']:
        train_examples=dataset['train']
        valid_examples=dataset['test']
    elif task_name in ['wiki_bio']:
        train_examples=dataset['train']
        valid_examples=dataset['val']
    elif task_name.split('/')[0]=='story_cloze':
        train_examples=None
        valid_examples=dataset['validation']
    else:
        train_examples=dataset['train']
        valid_examples=dataset['validation']
    if k_fold!=-1 and fold_id!=-1:
        cached_path=os.path.join(data_dir,'cached_shuffled_idxs','{}_fold{}of{}.npy'.format(task_prompt_name,k_fold,fold_id))
        cached_fold_path=os.path.join(data_dir,'cached_shuffled_idxs','{}_fold{}of{}.npy'.format(task_name,k_fold,fold_id))
        if os.path.exists(cached_path):
            matched_idxs=np.load(cached_path)
            print(cached_path,'load_cached')
        else:
            if os.path.exists(cached_fold_path):
                fold_idxs=np.load(cached_fold_path)
            else:
                index_path=os.path.join(data_dir,'shuffled_idxs','{}.npy'.format(task_name))
                # print('start load shuffled idxes')
                shuffled_idxs=np.load(index_path)
                # print('finished load shuffled idxes')
                each_fold_num=len(shuffled_idxs)//k_fold
                fold_idxs=shuffled_idxs[each_fold_num*fold_id:each_fold_num*(fold_id+1)]
                fold_idxs.sort()
                if os.path.exists('/'.join(cached_fold_path.split('/')[:-1]))==False:
                    os.makedirs('/'.join(cached_fold_path.split('/')[:-1]))
                np.save(cached_fold_path,fold_idxs)
            tmp_idxs=[e['idx'] for e in train_examples]
            matched_idxs=[]
            print(len(tmp_idxs),len(fold_idxs))
            i=0;j=0
            while(i<len(tmp_idxs) and j<len(fold_idxs)):
                if tmp_idxs[i]==fold_idxs[j]:
                    matched_idxs.append(i);i+=1;j+=1;
                elif tmp_idxs[i]>fold_idxs[j]:
                    j+=1
                else:
                    i+=1
            np.save(cached_path,matched_idxs)
        train_examples=train_examples.select(matched_idxs)
    if keep_origin==True:
        return train_examples,valid_examples

    # if the number of examples is larger than the required number, just cut it.
    # print('before', len(train_examples))
    if train_examples is not None and train_num!=-1 and len(train_examples)>train_num:
        train_idxes=rng.sample(range(len(train_examples)),train_num//prompt_num)
        train_examples=train_examples.select(train_idxes)

    if valid_examples is not None and valid_num!=-1 and len(valid_examples)>valid_num:
        valid_idxes=rng.sample(range(len(valid_examples)),valid_num//prompt_num)
        valid_examples=valid_examples.select(valid_idxes)
    return train_examples, valid_examples

class MultiPromptDataset(Dataset):
    def __init__(self,inputs,targets,task_names=None,prompt_names=None):
        self.inputs=inputs
        self.targets=targets
        self.task_names=task_names if task_names is not None and len(task_names)>0 else None
        self.prompt_names=prompt_names if prompt_names is not None and len(prompt_names)>0 else None

    def __len__(self,):
        return len(self.targets)

    def __getitem__(self,idx):
        ret_e={'inputs_pretokenized':self.inputs[idx],
                'targets_pretokenized':self.targets[idx]
                }
        if self.prompt_names is not None:
            ret_e['prompt_name']=self.prompt_names[idx]
        if self.task_names is not None:
            ret_e['task_name']=self.task_names[idx]
        return ret_e

def load_and_cache_examples_from_p3(args,task_names):
    inputs={'train':[],'valid':[]};targets={'train':[],'valid':[]}
    for task_name in task_names:
        print_rank_0('preparing dataset {}'.format(task_name))
        data_dir=os.path.join(args.multi_cache_dir,task_name)
        examples={'train':None,'valid':None}
        prompts = DatasetTemplates(task_name)
        for set_type in ['train','valid']:
            total_prompt_names=[x for x in prompts.name_to_id_mapping]
            prompt_num=len(total_prompt_names)         
            for prompt_name in total_prompt_names:
                data_dir=os.path.join(args.multi_cache_dir,'P3')
                examples['train'],examples['valid']=load_datasets_p3(data_dir=data_dir,task_name=task_name,prompt_name=prompt_name,train_num=args.max_train_num_per_dataset,valid_num=args.max_valid_num_per_dataset,prompt_num=prompt_num,k_fold=args.k_fold,fold_id=args.fold_id)
                if examples[set_type] is None: continue
                inputs[set_type]+=examples[set_type]['inputs_pretokenized']
                targets[set_type]+=examples[set_type]['targets_pretokenized']
                if set_type=='train': print(len(examples['train']))
    train_dataset=MultiPromptDataset(inputs['train'],targets['train'])
    valid_dataset=MultiPromptDataset(inputs['valid'],targets['valid'])
    return train_dataset,valid_dataset

def load_and_cache_examples_from_origdataset(args,task_names):
    '''
    set_type: ['train','test']
    return_tensors: return tensors or MultiPromptDataset
    '''
    inputs={'train':[],'valid':[]};targets={'train':[],'valid':[]}
    for task_name in task_names:
        logger.info('preparing dataset {}'.format(task_name))
        train_examples,valid_examples=load_datasets(data_dir=args.multi_cache_dir,task_name=task_name,train_num=args.max_train_num_per_dataset,valid_num=args.max_valid_num_per_dataset,k_fold=args.k_fold,fold_id=args.fold_id)
        total_examples={'train':train_examples,'valid':valid_examples}
        prompts = DatasetTemplates(task_name)
        for set_type in ['train','valid']:
            examples=total_examples[set_type]
            task_inputs=[];task_targets=[]
            total_prompt_names=[x for x in prompts.name_to_id_mapping]
            for prompt_name in total_prompt_names:
                prompt=prompts[prompt_name]
                for e in tqdm(examples):
                    prompt_ans=prompt.apply(e)
                    if len(prompt_ans)==1 or len(prompt_ans[1])==0: continue
                    input_tokens,target=prompt_ans
                    task_inputs.append(input_tokens)
                    task_targets.append(target)
            if len(task_inputs)!=0:
                inputs[set_type]+=task_inputs;targets[set_type]+=task_targets
    train_dataset=MultiPromptDataset(inputs['train'],targets['train'])
    valid_dataset=MultiPromptDataset(inputs['valid'],targets['valid'])
    return train_dataset,valid_dataset

def load_and_cache_examples(args,task_names,from_p3=True):
    if from_p3:
        return load_and_cache_examples_from_p3(args,task_names)
    else:
        return load_and_cache_examples_from_origdataset(args,task_names)

if __name__ == '__main__':
    from universal_da.dataset_config import default_T0_tasks
    from arguments import get_args
    args=get_args()
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.rank in [-1, 0] else logging.WARN,
    )
    # import pdb 
    # pdb.set_trace()
    task_names=[]
    for task_type,names in default_T0_tasks.items():
        task_names+=names
    # task_names=['coqa','wiki_hop/original','wiki_bio','samsum']
    # task_names=['super_glue/rte']
    train_dataset,valid_dataset=load_and_cache_examples(args,task_names,from_p3=False)
    import pdb 
    pdb.set_trace()



'''
CUDA_VISIBLE_DEVICES=7 python -m universal_da.T0_data_utils --multi-cache-dir ../../huggingface_datasets

CUDA_VISIBLE_DEVICES=7 python -m universal_da.T0_data_utils --multi-cache-dir ../../huggingface_datasets --k_fold 2 --fold_id 0
'''

