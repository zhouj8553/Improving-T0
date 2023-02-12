# TODO: only paraphrase is based on the para1, else based on the original paragraph
import os
import re
import math
import string
import random
import copy
from timeit import default_timer
import torch
import numpy as np
from tqdm import tqdm
import datasets
import json
from scipy.special import softmax
from multiprocessing import Pool

from SwissArmyTransformer.model import T5Model
from universal_da.simple_cross import convert_dataset_uniform
from universal_da.simple_cross.uniform_prompts import uniformed_prompt_templates
from universal_da.dataset_config import default_T0_tasks
from universal_da.simple_cross.filter_cross_data import build_uniformed_dataset_from_crossdatasets_0527
from nltk.tokenize import sent_tokenize
def get_select_pos(predictions,th=0.1,topk=-1):
    if len(predictions)==0: return []
    predictions=np.array(predictions)
    if int(th*len(predictions))==len(predictions): return [True]*len(predictions)
    sorted_idx=np.argsort(-predictions)
    if topk!=-1:
        threshold_idx=sorted_idx[min(topk-1,len(sorted_idx)-1)]
    else:
        threshold_idx=sorted_idx[int(th*len(predictions))]
    threshold_value=predictions[threshold_idx]
    select_pos=(np.array(predictions)>=threshold_value)
    return select_pos

def filter_paraphrase_dataset_by_prob(args,task_name,uniformed_dataset,th=0.1):
    ######################################################################################################
    # output: [{'para1':'I like NLP.','para1_meta':{'similar_sentence':['I like math.'],'paraphrase_label':[0]}}]
    # step1: group by labels
    # step2: select the most possible examples for each group
    ######################################################################################################
    candidate_groups=dict()
    candidate_group_logits=dict()
    for e in uniformed_dataset:
        # import pdb 
        # pdb.set_trace()
        for (sentence,label) in zip(e['para1_meta']['similar_sentence'],e['para1_meta']['paraphrase_label']):
            if label[0] not in candidate_groups:
                candidate_groups[label[0]]=[]
                candidate_group_logits[label[0]]=[]
            candidate_groups[label[0]].append({'para1':e['para1'][0],'para1_meta':{'similar_sentence':[sentence[0]],'paraphrase_label':[label[0]]}})
            candidate_group_logits[label[0]].append(label[1])
    # import pdb 
    # pdb.set_trace()
    selected_examples=[]
    neg_num=len(candidate_group_logits[0])
    pos_num=len(candidate_group_logits[1])
    min_num=int(min(neg_num,pos_num)*th)
    for (group_id,logits) in candidate_group_logits.items():
        select_pos=get_select_pos(logits,topk=min_num)
        selected_examples+=[x for (x,is_selected) in zip(candidate_groups[group_id],select_pos) if is_selected==True]
    # import pdb 
    # pdb.set_trace()
    return selected_examples


def filter_paraphrase_dataset_by_example(args,task_name,uniformed_dataset,th=0.1):
    ######################################################################################################
    # output: [{'para1':'I like NLP.','para1_meta':{'similar_sentence':['I like math.'],'paraphrase_label':[0]}}]
    # step1: group by labels
    # step2: select the most possible examples for each group
    ######################################################################################################
    selected_examples=[]
    for e in uniformed_dataset:
        candidate_groups=dict()
        candidate_group_logits=dict()
        for (sentence,label) in zip(e['para1_meta']['similar_sentence'],e['para1_meta']['paraphrase_label']):
            if label[0] not in candidate_groups:
                candidate_groups[label[0]]=[]
                candidate_group_logits[label[0]]=[]
            if task_name=='gigaword' and e['para1'][0] is None: 
                para1=e['para2'][0]
            else:
                para1=e['para1'][0]
            candidate_groups[label[0]].append({'para1':para1,'para1_meta':{'similar_sentence':[sentence[0]],'paraphrase_label':[label[0]]}})
            candidate_group_logits[label[0]].append(label[1])
        for label in [0,1]:
            if label not in candidate_groups or len(candidate_groups[label])==0: continue
            idx=np.argmax(candidate_group_logits[label])
            selected_examples.append(candidate_groups[label][idx])
    return selected_examples

def filter_paraphrase_dataset(args,task_name,uniformed_dataset,th=0.1):
    return filter_paraphrase_dataset_by_example(args,task_name,uniformed_dataset,th)

def filter_sentiment2_dataset(args,task_name,uniformed_dataset,th=0.1,prob_th=-1,with_prob=False):
    ######################################################################################################
    # output: [{'para1':'I like NLP.','attributes':{'sentiment_5':{'answer':0}}}]
    # step1: group by labels
    # step2: select the most possible examples for each group
    ######################################################################################################
    candidate_groups={0:[],1:[]}
    candidate_group_logits={0:[],1:[]}
    for (i,e) in enumerate(uniformed_dataset):
        label=e['attributes']['sentiment_2']['answer']
        para=e['para1'][0] if (e['para1'][1]==100 and e['para1'][0]!=None) else e['para2'][0]
        if with_prob==True:
            new_e={'para1':para,'attributes':{'sentiment_2':{'answer':label}}}
        else:
            new_e={'para1':para,'attributes':{'sentiment_2':{'answer':label[0]}}}
        # new_e['attributes']['title']['answer']=e['attributes']['title']['answer'][0]
        if 'attributes' in new_e:
            new_e['attributes']['title']={'answer':e['attributes']['title']['answer'][0]}
        else:
            new_e['attributes']={'title':{'answer':e['attributes']['title']['answer'][0]}}
        # if 'topic1' in la_name: new_e['attributes']['topic1']['candidates']=e['attributes']['topic1']['candidates']
        # if 'topic2' in la_name: new_e['attributes']['topic2']['candidates']=e['attributes']['topic2']['candidates']
        if label[0]==0 and (e['attributes']['sentiment_5']['answer'][0] is not None and e['attributes']['sentiment_5']['answer'][0] in [3,4,5]):
            continue
        if label[0]==1 and (e['attributes']['sentiment_5']['answer'][0] is not None and e['attributes']['sentiment_5']['answer'][0] in [1,2,3]):
            continue
        if label[0] is not None and label[1]>=prob_th:
            candidate_groups[label[0]].append(new_e)
            candidate_group_logits[label[0]].append(label[1])
    selected_examples=[]
    for (group_id,logits) in candidate_group_logits.items():
        select_pos=get_select_pos(logits,th=th)
        selected_examples+=[x for (x,is_selected) in zip(candidate_groups[group_id],select_pos) if is_selected==True]
    return selected_examples

def filter_sentiment5_dataset(args,task_name,uniformed_dataset,th=0.1,prob_th=-1,with_prob=False):
    ######################################################################################################
    # output: [{'para1':'I like NLP.','attributes':{'sentiment_5':{'answer':0}}}]
    # step1: group by labels
    # step2: select the most possible examples for each group
    ######################################################################################################
    candidate_groups={1:[],2:[],3:[],4:[],5:[]}
    candidate_group_logits={1:[],2:[],3:[],4:[],5:[]}
    for (i,e) in enumerate(uniformed_dataset):
        label=e['attributes']['sentiment_5']['answer']
        para=e['para1'][0] if (e['para1'][1]==100 and e['para1'][0]!=None) else e['para2'][0]
        if with_prob==True:
            new_e={'para1':para,'attributes':{'sentiment_5':{'answer':label}}}
        else:
            new_e={'para1':para,'attributes':{'sentiment_5':{'answer':label[0]}}}
        # new_e['attributes']['title']['answer']=e['attributes']['title']['answer'][0]
        if 'attributes' in new_e:
            new_e['attributes']['title']={'answer':e['attributes']['title']['answer'][0]}
        else:
            new_e['attributes']={'title':{'answer':e['attributes']['title']['answer'][0]}}
        # if 'topic1' in la_name: new_e['attributes']['topic1']['candidates']=e['attributes']['topic1']['candidates']
        # if 'topic2' in la_name: new_e['attributes']['topic2']['candidates']=e['attributes']['topic2']['candidates']
        if label[0] in [1,2] and (e['attributes']['sentiment_2']['answer'][0] is not None and e['attributes']['sentiment_2']['answer'][0]==1):
            continue
        if label[0] in [4,5] and (e['attributes']['sentiment_2']['answer'][0] is not None and e['attributes']['sentiment_2']['answer'][0]==0):
            continue
        if label[0] is not None and label[1]>=prob_th:
            candidate_groups[label[0]].append(new_e)
            candidate_group_logits[label[0]].append(label[1])
    selected_examples=[]
    for (group_id,logits) in candidate_group_logits.items():
        select_pos=get_select_pos(logits,th=th)
        selected_examples+=[x for (x,is_selected) in zip(candidate_groups[group_id],select_pos) if is_selected==True]
    return selected_examples

# filter_cls_dataset(args,task_name,uniformed_dataset,la_name=['attributes','topic_1','answer'])
# filter_cls_dataset(args,task_name,uniformed_dataset,la_name=['attributes','topic_2','answer'])
def filter_cls_dataset(args,task_name,uniformed_dataset,th=0.1,la_name=['attributes','sentiment_2','answer'],prob_th=-1,with_prob=False):
    ######################################################################################################
    # output: [{'para1':'I like NLP.','attributes':{'sentiment_5':{'answer':0}}}]
    # step1: group by labels
    # step2: select the most possible examples for each group
    ######################################################################################################
    candidate_groups={}
    candidate_group_logits={}
    for (i,e) in enumerate(uniformed_dataset):
        label=e
        for name in la_name: label=label[name]
        try:
            if label[0] not in candidate_groups:
                candidate_groups[label[0]]=[]
                candidate_group_logits[label[0]]=[]
        except:
            import pdb 
            pdb.set_trace()
        para=e['para1'][0] if (e['para1'][1]==100 and e['para1'][0]!=None) else e['para2'][0]
        new_e={'para1':para}
        label_dict=label[0] if with_prob==False else label
        for name in la_name[::-1]: label_dict={name: label_dict}
        new_e=dict(**new_e,**label_dict)
        # new_e['attributes']['title']['answer']=e['attributes']['title']['answer'][0]
        if 'attributes' in new_e:
            new_e['attributes']['title']={'answer':e['attributes']['title']['answer'][0]}
        else:
            new_e['attributes']={'title':{'answer':e['attributes']['title']['answer'][0]}}
        if 'topic1' in la_name: new_e['attributes']['topic1']['candidates']=e['attributes']['topic1']['candidates']
        if 'topic2' in la_name: new_e['attributes']['topic2']['candidates']=e['attributes']['topic2']['candidates']
        if label[1]>=prob_th:
            candidate_groups[label[0]].append(new_e)
            candidate_group_logits[label[0]].append(label[1])
    selected_examples=[]
    for (group_id,logits) in candidate_group_logits.items():
        select_pos=get_select_pos(logits,th=th)
        selected_examples+=[x for (x,is_selected) in zip(candidate_groups[group_id],select_pos) if is_selected==True]
    return selected_examples

def filter_gen_dataset(args,task_name,uniformed_dataset,th=0.1,la_name=['attributes','keywords','answer']):
    ######################################################################################################
    # output: [{'para1':'I like NLP.','attributes':{'title':{'answer':'NLP'}}}]
    # step1: group by labels
    # step2: select the most possible examples for each group
    ######################################################################################################
    logits=[]
    candidate_examples=[]
    for e in uniformed_dataset:
        para=e['para1'][0] if (e['para1'][1]==100 and e['para1'][0]!=None) else e['para2'][0]
        new_e={'para1':para}
        if 'keywords' in la_name:
            new_e=dict(**new_e,**{'attributes':{'keywords':{'answer':[x for (x,y) in e['attributes']['keywords']['answer']]}}})
            logit=(np.mean([y for (x,y) in e['attributes']['keywords']['answer']]))
        else:
            label=e 
            for name in la_name: label=label[name]
            label_dict=label[0]
            for name in la_name[::-1]: label_dict={name: label_dict}
            new_e=dict(**new_e,**label_dict)
            logit=(e['attributes']['title']['answer'][1])
        if 'attributes' in new_e:
            new_e['attributes']['title']={'answer':e['attributes']['title']['answer'][0]}
        else:
            new_e['attributes']={'title':{'answer':e['attributes']['title']['answer'][0]}}
        candidate_examples.append(new_e)
        logits.append(logit)
    # import pdb 
    # pdb.set_trace()
    select_pos=get_select_pos(logits,th=th)
    selected_examples=[x for (x,is_selected) in zip(candidate_examples,select_pos) if is_selected==True]
    return selected_examples

def filter_summary_dataset(args,task_name,uniformed_dataset,th=0.1):
    ######################################################################################################
    # output: [{'para1':'I like NLP.','para2':'I like NLP, because it is very very interesting.'}]
    # step1: group by labels
    # step2: select the most possible examples for each group
    ######################################################################################################  
    logits=[]
    candidate_examples=[]
    for e in uniformed_dataset:
        # new_e={'para1':e['para1'][0],'para2':e['para2'][0]}
        if task_name=='gigaword': new_e={'para1':e['attributes']['title']['answer'][0],'para2':e['para2'][0]}
        else: new_e={'para1':e['para1'][0],'para2':e['para2'][0]}
        candidate_examples.append(new_e)
        logits.append(e['para1'][1]+e['para2'][1])
    select_pos=get_select_pos(logits,th=th)
    selected_examples=[x for (x,is_selected) in zip(candidate_examples,select_pos) if is_selected==True]
    return selected_examples

def filter_qa_dataset(args,task_name,uniformed_dataset,th=0.1):
    # target_type could be 'open', 'multi'
    # only keep wrong examples most confident (hard examples)
    candidate_examples=[];logits=[]
    for e in uniformed_dataset:
        # only choose one (question, answer) pair if several questions exists
        if len(e['questions'])==0: continue
        question_idx=np.random.choice(range(len(e['questions']))) if len(e['questions'])>1 else 0
        # print('sess',len(e['questions']),question_idx)
        question=e['questions'][question_idx]
        answer=e['answers'][question_idx]
        # for (question, answer) in zip(e['questions'],e['answers']):
            # print(question,answer)
        para=e['para1'][0] if (e['para1'][1]==100 and e['para1'][0]!=None) else e['para2'][0]
        new_e={'para1':para,'question':question[0],'attributes':{'title':{'answer':e['attributes']['title']['answer'][0]}}}
        correct_answers=[x[0] for x in answer['correct']]
        wrong_answers=[x[0] for x in answer['wrong']]
        wrong_logits=[x[1] for x in answer['wrong']]
        select_wrong_pos=get_select_pos([-x for x in wrong_logits],topk=3)
        selected_wrong_answer=[x for (x,is_selected) in zip(wrong_answers,select_wrong_pos) if is_selected==True]
        new_e=dict(**new_e,**{'answers':{'correct':correct_answers,'wrong':selected_wrong_answer}})
        candidate_examples.append(new_e)
        logits.append(answer['correct'][0][1])
    # import pdb 
    # pdb.set_trace()
    assert len(candidate_examples)==len(logits)
    select_pos=get_select_pos(logits,th=th)
    selected_examples=[x for (x,is_selected) in zip(candidate_examples,select_pos) if is_selected==True]
    return selected_examples

def filter_doc_dataset(args,task_name,uniformed_dataset,th=0.1):
    # choose the longest sentences
    candidate_examples=[];logits=[]
    for e in uniformed_dataset:
        para=e['para2'][0] if e['para2'][1]==100 else e['para1'][0]
        if para is None: continue
        if '\n\n' in para: 
            para_list=para.split('\n\n')
            if len(para_list)==1: continue
            logits.append(len(para_list))
            new_e={'para1':para}
        else: 
            para_list=sent_tokenize(para)
            para='\n\n'.join(para_list).strip('\n\n')
            if '\n\n' not in para: continue
            logits.append(len(para_list))
            new_e={'para1':para}
        candidate_examples.append(new_e)
    select_pos=get_select_pos(logits,th=th)
    selected_examples=[x for (x,is_selected) in zip(candidate_examples,select_pos) if is_selected==True]
    return selected_examples


def get_sub_qa_dataset_per(input):
    [e,question_type,limit_num]=input
    (limit_lower_bound,limit_upper_bound)=limit_num
    new_e=copy.deepcopy(e)
    answers=new_e.pop('answers')
    if question_type=='multi_choice' and len(answers['wrong'])==0: return None
    if len(answers['wrong'])<limit_lower_bound: return None
    if len(answers['correct'])==0: answers['correct'].append('no answer')
    choosen_answer=np.random.choice(answers['correct'])
    choices=[choosen_answer]+np.random.choice(answers['wrong'],min(limit_upper_bound,len(answers['wrong'])),False).tolist()
    random.shuffle(choices)
    new_e['choices']=choices
    if question_type!='tf':
        new_e['answer']=choosen_answer
        new_e['answer_label']=1
        return new_e
    else:
        ans={'True':None,'False':None}
        if len(answers['correct'])!=0:
            nnew_e=copy.deepcopy(new_e)
            nnew_e['answer']=choosen_answer
            nnew_e['answer_label']=1
            # ret_examples['True'].append(nnew_e)
            ans['True']=nnew_e
        if len(answers['wrong'])!=0:
            nnew_e=copy.deepcopy(new_e)
            nnew_e['answer']=np.random.choice(answers['wrong'])
            nnew_e['answer_label']=0
            # ret_examples['False'].append(nnew_e)
            ans['False']=nnew_e
        return ans

def get_sub_qa_dataset(examples,question_type='tf',limit_num=(0,100),parallel=False):
    # question_type: choose from ['tf','gen','multi_choice']
    ret_examples={'True':[],'False':[]} if question_type=='tf' else []
    results=[]
    if parallel==True:
        pool=Pool(processes=1)
        results=pool.map(get_sub_qa_dataset_per,[(e,question_type,limit_num) for e in examples])
        pool.close()
    else:
        results=[get_sub_qa_dataset_per((e,question_type,limit_num)) for e in examples]
    if question_type!='tf':
        ret_examples=[e for e in results if e is not None]
    else:
        for res in results:
            if res['True'] is not None: ret_examples['True'].append(res['True'])
            if res['False'] is not None: ret_examples['False'].append(res['False'])
        minlen=min(len(ret_examples['True']),len(ret_examples['False']))
        true_examples=np.random.choice(ret_examples['True'],minlen,False)
        false_examples=np.random.choice(ret_examples['False'],minlen,False)
        ret_examples=true_examples.tolist()+false_examples.tolist()
    # import pdb 
    # pdb.set_trace()
    return ret_examples

def apply_prompt(x):
    (example,prompt)=x
    try:
        result=prompt.apply(example)
        if len(result)!=2 or result[1] is None or result[1]=='None' or len(result[1])==0:
            return None
    except:
        result=None
    return result

def apply_prompts(examples,prompts,parallel=False):
    prompted_datas=[]
    if parallel==True:
        pool=Pool(processes=1)
        print(len(examples),len(prompts))
        results=pool.map(apply_prompt,[(e,prompt) for e in tqdm(examples) for prompt in prompts])
        pool.close()
    else:
        results=[apply_prompt((e,prompt)) for e in examples for prompt in prompts]
    for result in results:
    # for e in tqdm(examples):
    #     result=prompt.apply(e)
        if result is None or len(result)<2 or result[1] is None or result[1]=='None' or len(result[1])==0: 
            continue
        prompted_datas.append({'input_tokens':result[0],'label':result[1]})
    prompted_datas=[{'input_tokens':result[0],'label':result[1]} for result in results if (result is not None)]
    return prompted_datas

def get_prompted_candidates(args,task_name,dataset=None):
    if dataset is None:
        dataset=dict()
        uniformed_dataset=build_uniformed_dataset_from_crossdatasets(args,task_name)
        dataset['paraphrase']=filter_paraphrase_dataset(args,task_name,uniformed_dataset)
        dataset['sentiment_2']=filter_cls_dataset(args,task_name,uniformed_dataset,la_name=['attributes','sentiment_2','answer'])
        dataset['sentiment_5']=filter_cls_dataset(args,task_name,uniformed_dataset,la_name=['attributes','sentiment_5','answer'])
        dataset['topic1']=filter_cls_dataset(args,task_name,uniformed_dataset,la_name=['attributes','topic1','answer'])
        dataset['topic2']=filter_cls_dataset(args,task_name,uniformed_dataset,la_name=['attributes','topic2','answer'])
        dataset['keywords']=filter_gen_dataset(args,task_name,uniformed_dataset,la_name=['attributes','keywords','answer'])
        dataset['title']=filter_gen_dataset(args,task_name,uniformed_dataset,la_name=['attributes','title','answer'])
        dataset['summary']=filter_summary_dataset(args,task_name,uniformed_dataset)
        dataset['qa']=filter_qa_dataset(args,task_name,uniformed_dataset)
        dataset['doc']=filter_doc_dataset(args,task_name,uniformed_dataset)
    else:
        dataset=dataset
    # import pdb 
    # pdb.set_trace()
    prompted_datas=dict()
    remain_to_be_solved=['question_paraphrase_tf']
    for (subtask_name,examples) in dataset.items():
        # import pdb 
        # pdb.set_trace()
        if subtask_name=='doc':
            prompts=[y for (x,y) in uniformed_prompt_templates['paragraph_to_paragraph'].items()]
            # import pdb 
            # pdb.set_trace()
            prompted_datas[subtask_name]=apply_prompts(examples,prompts)
        elif subtask_name=='paraphrase':
            prompts=[y for (x,y) in uniformed_prompt_templates['generate_paraphrased_sentence'].items()]
            prompted_datas[subtask_name]=apply_prompts([e for e in examples if e['para1_meta']['paraphrase_label'][0]==1],prompts)
            prompts=[y for (x,y) in uniformed_prompt_templates['paragraph_question_tf'].items()]
            prompted_datas[subtask_name]+=apply_prompts(examples,prompts)
            # import pdb 
            # pdb.set_trace()
        elif subtask_name=='qa':
            # qa_closed_book
            for qa_type in ['qa_closed_book','qa_extractive','qa_multiple_choice']:
                limit_lower_bound=0
                if qa_type=='qa_closed_book':
                    prompt_names=['question_to_answer','answer_title_to_question']+['question_answer_to_title','question_to_title','answer_to_title']
                    question_type='gen'
                elif qa_type=='qa_extractive':
                    prompt_names=['paragraph_hints_question_to_answer','paragraph_question_to_answer','paragraph_question_title_to_answer','paragraph_to_question','paragraph_answer_to_question','question_answer_to_paragraph']+['question_answer_to_paragraph']
                    question_type='gen'
                elif qa_type=='qa_multiple_choice':
                    prompt_names=['question_to_choose_answer','paragraph_question_to_choose_answer','paragraph_hints_question_to_choose_answer']
                    question_type='multi_choice'
                    limit_lower_bound=1
                prompts=sum([[y for (x,y) in uniformed_prompt_templates[prompt_name].items()] for prompt_name in prompt_names],[])
                prompted_datas[qa_type]=[]
                for (prompt_name,prompt) in zip(prompt_names,prompts):
                    if prompt_name in ['do_not_use','logic_test','heres_a_story','choose_between','testing_students','Multiple Choice (Closed Book)']:
                        limit_upper_bound=1
                    else: limit_upper_bound=100
                    sub_examples=get_sub_qa_dataset(examples,question_type=question_type,limit_num=(limit_lower_bound,limit_upper_bound))
                    prompted_datas[qa_type]+=apply_prompts(sub_examples,[prompt])
            prompt_names=['question_to_answer_tf','title_question_to_answer_tf','paragraph_question_to_answer_tf']
            prompts=sum([[y for (x,y) in uniformed_prompt_templates[prompt_name].items()] for prompt_name in prompt_names],[])
            prompted_data['qa_tf']=[]
            for (prompt_name,prompt) in zip(prompt_names,prompts):
                sub_examples=get_sub_qa_dataset(examples,question_type='tf')
                prompted_datas['qa_tf']+=apply_prompts(sub_examples,prompts)
        elif subtask_name=='sentiment_2' or subtask_name=='sentiment_5':
            # Attention that the prompt of sentiment_2 and sentiment_5 is not divided.
            prompt_names=['paragraph_to_sentiment','paragraph_title_to_sentiment','sentiment_title_to_paragraph']
            prompts=sum([[y for (x,y) in uniformed_prompt_templates[prompt_name].items()] for prompt_name in prompt_names],[])
            prompted_datas[subtask_name]=apply_prompts(examples,prompts)
        elif subtask_name=='keywords':
            prompt_names=['keywords_to_paragraph','paragraph_to_keywords']
            prompts=sum([[y for (x,y) in uniformed_prompt_templates[prompt_name].items()] for prompt_name in prompt_names],[])
            prompted_datas[subtask_name]=apply_prompts(examples,prompts)
        elif subtask_name=='title':
            prompt_names=['paragraph_to_title']
            prompts=sum([[y for (x,y) in uniformed_prompt_templates[prompt_name].items()] for prompt_name in prompt_names],[])
            prompted_datas[subtask_name]=apply_prompts(examples,prompts)
        elif subtask_name=='summary':
            prompt_names=['paragraph3_to_paragraph1','paragraph1_to_paragraph3']
            prompts=sum([[y for (x,y) in uniformed_prompt_templates[prompt_name].items()] for prompt_name in prompt_names],[])
            prompted_datas[subtask_name]=apply_prompts(examples,prompts)
        elif subtask_name=='topic1':
            prompt_names=['paragraph_to_topic1']
            prompts=sum([[y for (x,y) in uniformed_prompt_templates[prompt_name].items()] for prompt_name in prompt_names],[])
            prompted_datas['topic1']=apply_prompts(examples,prompts)
        elif subtask_name=='topic2':
            prompt_names=['paragraph_to_topic2','title_to_topic2','paragraph_title_to_topic2']
            prompts=sum([[y for (x,y) in uniformed_prompt_templates[prompt_name].items()] for prompt_name in prompt_names],[])
            prompted_datas['topic2']=apply_prompts(examples,prompts)
        # import pdb 
        # pdb.set_trace()
        if subtask_name!='qa':
            if len(prompted_datas[subtask_name])!=0:
                print(prompted_datas[subtask_name][0])
        else:
            print(prompted_datas['qa_closed_book'][0])
            print(prompted_datas['qa_extractive'][0])
            print(prompted_datas['qa_multiple_choice'][0])
            print(prompted_datas['qa_tf'][0])
    return prompted_datas

import time
def get_prompted_candidates_onlyupdateqa(args,task_name,dataset=None,already_data=None):
    if dataset is None:
        dataset=dict()
        uniformed_dataset=build_uniformed_dataset_from_crossdatasets_0527(args,task_name)
        dataset['paraphrase']=filter_paraphrase_dataset(args,task_name,uniformed_dataset)
        dataset['sentiment_2']=filter_sentiment2_dataset(args,task_name,uniformed_dataset)
        dataset['sentiment_5']=filter_sentiment5_dataset(args,task_name,uniformed_dataset)
        dataset['topic1']=filter_cls_dataset(args,task_name,uniformed_dataset,la_name=['attributes','topic1','answer'])
        dataset['topic2']=filter_cls_dataset(args,task_name,uniformed_dataset,la_name=['attributes','topic2','answer'])
        dataset['keywords']=filter_gen_dataset(args,task_name,uniformed_dataset,th=0.1,la_name=['attributes','keywords','answer'])
        dataset['title']=filter_gen_dataset(args,task_name,uniformed_dataset,th=0.1,la_name=['attributes','title','answer'])
        dataset['summary']=filter_summary_dataset(args,task_name,uniformed_dataset)
        dataset['qa']=filter_qa_dataset(args,task_name,uniformed_dataset,th=1.0)
        # dataset['doc']=filter_doc_dataset(args,task_name,uniformed_dataset)
    else:
        dataset=dataset
    # import pdb 
    # pdb.set_trace()
    prompted_datas=dict()
    remain_to_be_solved=['question_paraphrase_tf']
    for (subtask_name,examples) in dataset.items():
        if subtask_name=='keywords':
            prompt_names=['keywords_to_paragraph','paragraph_to_keywords']
            prompts=sum([[y for (x,y) in uniformed_prompt_templates[prompt_name].items()] for prompt_name in prompt_names],[])
            prompted_datas[subtask_name]=apply_prompts(examples,prompts)
        elif subtask_name=='title':
            prompt_names=['paragraph_to_title']
            prompts=sum([[y for (x,y) in uniformed_prompt_templates[prompt_name].items()] for prompt_name in prompt_names],[])
            prompted_datas[subtask_name]=apply_prompts(examples,prompts)
        elif subtask_name=='sentiment_2':
            prompt_names=['paragraph_title_to_sentiment']
            prompts=sum([[y for (x,y) in uniformed_prompt_templates[prompt_name].items()] for prompt_name in prompt_names],[])
            prompts+=[uniformed_prompt_templates['paragraph_to_sentiment'][x] for x in ['User_recommend_this_product', 'Movie Expressed Sentiment 2', 'Reviewer Opinion bad good choices', 'Sentiment with choices ', 'Reviewer Sentiment Feeling', 'Writer Expressed Sentiment', 'Movie Expressed Sentiment', 'Text Expressed Sentiment', 'Negation template for positive and negative', 'Reviewer Enjoyment Yes No', 'Reviewer Expressed Sentiment','Reviewer Enjoyment']]
            prompted_datas[subtask_name]=apply_prompts(examples,prompts)
        elif subtask_name=='sentiment_5':
            prompt_names=['sentiment_title_to_paragraph']
            prompts=sum([[y for (x,y) in uniformed_prompt_templates[prompt_name].items()] for prompt_name in prompt_names],[])
            prompts+=[uniformed_prompt_templates['paragraph_to_sentiment'][x] for x in ['categorize_rating_using_review', 'convert_to_star_rating', 'convert_to_rating', 'so_i_would', 'based_on_that', 'format_star', 'this_place', 'format_score', 'on_a_scale', 'format_rating']]
            prompted_datas[subtask_name]=apply_prompts(examples,prompts)            
        elif subtask_name=='topic1':
            prompt_names=['paragraph_to_topic1']
            prompts=sum([[y for (x,y) in uniformed_prompt_templates[prompt_name].items()] for prompt_name in prompt_names],[])
            prompted_datas['topic1']=apply_prompts(examples,prompts)
        elif subtask_name=='topic2':
            prompt_names=['paragraph_to_topic2','title_to_topic2','paragraph_title_to_topic2']
            prompts=sum([[y for (x,y) in uniformed_prompt_templates[prompt_name].items()] for prompt_name in prompt_names],[])
            prompted_datas['topic2']=apply_prompts(examples,prompts)
        elif task_name=='gigaword' and subtask_name=='paraphrase':
            prompts=[y for (x,y) in uniformed_prompt_templates['generate_paraphrased_sentence'].items()]
            prompted_datas[subtask_name]=apply_prompts([e for e in examples if e['para1_meta']['paraphrase_label'][0]==1],prompts)
            prompts=[y for (x,y) in uniformed_prompt_templates['paragraph_question_tf'].items()]
            prompted_datas[subtask_name]+=apply_prompts(examples,prompts)
        elif subtask_name=='summary':
            prompt_names=['paragraph3_to_paragraph1','paragraph1_to_paragraph3']
            prompts=sum([[y for (x,y) in uniformed_prompt_templates[prompt_name].items()] for prompt_name in prompt_names],[])
            prompted_datas[subtask_name]=apply_prompts(examples,prompts)
        # elif subtask_name=='qa':
        #     # import pdb 
        #     # pdb.set_trace()
        #     # qa_closed_book
        #     for qa_type in ['qa_closed_book','qa_extractive','qa_multiple_choice']:
        #         limit_lower_bound=0
        #         if qa_type=='qa_closed_book':
        #             prompt_names=['question_to_answer','answer_title_to_question']+['question_answer_to_title','question_to_title','answer_to_title']
        #             question_type='gen'
        #         elif qa_type=='qa_extractive':
        #             prompt_names=['paragraph_hints_question_to_answer','paragraph_question_to_answer','paragraph_question_title_to_answer','paragraph_to_question','paragraph_answer_to_question','question_answer_to_paragraph']+['question_answer_to_paragraph']
        #             question_type='gen'
        #         elif qa_type=='qa_multiple_choice':
        #             # prompt_names=['question_to_choose_answer','paragraph_question_to_choose_answer','paragraph_hints_question_to_choose_answer']
        #             prompt_names=['paragraph_question_to_choose_answer','paragraph_hints_question_to_choose_answer']
        #             question_type='multi_choice'
        #             limit_lower_bound=1
        #         prompts=sum([[y for (x,y) in uniformed_prompt_templates[prompt_name].items()] for prompt_name in prompt_names],[])
        #         prompted_datas[qa_type]=[]
        #         for prompt in prompts:
        #             prompt_name=prompt.name
        #             if prompt_name in ['do_not_use','logic_test','heres_a_story','choose_between','testing_students','Multiple Choice (Closed Book)']:
        #                 limit_upper_bound=1
        #             else: limit_upper_bound=100
        #             sub_examples=get_sub_qa_dataset(examples,question_type=question_type,limit_num=(limit_lower_bound,limit_upper_bound))
        #             print(prompt_name,len(sub_examples))
        #             # time2=time.time()
        #             prompted_datas[qa_type]+=apply_prompts(sub_examples,[prompt])
        #             # time3=time.time()
        #             # print(time2-time1,time3-time2)
        #             # pdb.set_trace()
        #     # prompt_names=['question_to_answer_tf','title_question_to_answer_tf','paragraph_question_to_answer_tf']
        #     prompt_names=['paragraph_question_to_answer_tf']
        #     prompts=sum([[y for (x,y) in uniformed_prompt_templates[prompt_name].items()] for prompt_name in prompt_names],[])
        #     prompted_datas['qa_tf']=[]
        #     for prompt in prompts:
        #         sub_examples=get_sub_qa_dataset(examples,question_type='tf')
        #         print(prompt.name,len(sub_examples))
        #         prompted_datas['qa_tf']+=apply_prompts(sub_examples,[prompt])
        else:
            if subtask_name=='qa':
                prompted_datas['qa_closed_book']=already_data['qa_closed_book']
                prompted_datas['qa_extractive']=already_data['qa_extractive']
                prompted_datas['qa_multiple_choice']=already_data['qa_multiple_choice']
                prompted_datas['qa_tf']=already_data['qa_tf']
            else:
                prompted_datas[subtask_name]=already_data[subtask_name]

        if subtask_name!='qa':
            if len(prompted_datas[subtask_name])!=0:
                print(prompted_datas[subtask_name][0])
        else:
            if len(prompted_datas['qa_closed_book'])!=0:
                print(prompted_datas['qa_closed_book'][0])
                print(prompted_datas['qa_extractive'][0])
                print(prompted_datas['qa_multiple_choice'][0])
                print(prompted_datas['qa_tf'][0])
    return prompted_datas

task_type_map={
    'paraphrase':['glue/mrpc','glue/qqp','paws/labeled_final'],
    'qa_closed_book':['kilt_tasks/hotpotqa','wiki_qa'],
    'qa_extractive':['adversarial_qa/dbidaf','adversarial_qa/dbert','adversarial_qa/droberta','duorc/SelfRC','duorc/ParaphraseRC','ropes','quoref'],
    'qa_multiple_choice':['cos_e/v1.11','cosmos_qa','dream','qasc','quail','quarel','quartz','sciq','social_i_qa','wiki_hop/original','wiqa'],
    'sentiment_2':['amazon_polarity','imdb','rotten_tomatoes'],
    'sentiment_5':['app_reviews','yelp_review_full'],
    'sentiment':['amazon_polarity','app_reviews','imdb','rotten_tomatoes','yelp_review_full'],
    'keywords':['common_gen','wiki_bio'],
    'summarization':['cnn_dailymail/3.0.0','gigaword','multi_news','samsum','xsum'],
    'topic1':['ag_news'],
    'topic2':['dbpedia_14'],
    'topic_question':['trec'],
}


def test_cls_tasks(uniformed_dataset,task_name):
    from collections import Counter
    sentiment2={None:[],0:[],1:[]}
    sentiment5={None:[],1:[],2:[],3:[],4:[],5:[]}
    topic1={None:[],'World politics': [], 'Business': [], 'Science and technology': [], 'Sports': []}
    topic2={None:[],'company': [], 'educational institution': [], 'artist': [], 'athlete': [], 'office holder': [], 'mean of transportation': [], 
    'building': [], 'natural place': [], 'village': [], 'animal': [], 'plant': [], 'album': [], 'film': [], 'written work': []}
    # imoprt pdb 
    # pdb.set_trace()
    for e in uniformed_dataset:
        # print(e)
        sentiment2[e['attributes']['sentiment_2']['answer'][0]].append(e['attributes']['sentiment_2']['answer'][1])
        sentiment5[e['attributes']['sentiment_5']['answer'][0]].append(e['attributes']['sentiment_5']['answer'][1])
        topic1[e['attributes']['topic1']['answer'][0]].append(e['attributes']['topic1']['answer'][1])
        topic2[e['attributes']['topic2']['answer'][0]].append(e['attributes']['topic2']['answer'][1])
    try:
        print('###################### sentiment2 ########################')
        for (x,y) in sentiment2.items():
            if x==None: continue
            if len(y)==0:
                print(task_name,x,'num: 0')
            else:
                print(task_name,x,'num: {}, mean: {}, percentile 50: {}, percentile 80: {}, percentile 90: {}'.format(len(y),np.mean(y),round(np.percentile(y,50),2),round(np.percentile(y,80),2),round(np.percentile(y,90),2)))
        
        print('###################### sentiment5 ########################')
        for (x,y) in sentiment5.items():
            if x==None: continue
            if len(y)==0:
                print(task_name,x,'num: 0')
            else:
                print(task_name,x,'num: {}, mean: {}, percentile 50: {}, percentile 80: {}, percentile 90: {}'.format(len(y),np.mean(y),round(np.percentile(y,50),2),round(np.percentile(y,80),2),round(np.percentile(y,90),2)))
        
        print('###################### topic1 ########################')
        for (x,y) in topic1.items():
            if x==None: continue
            if len(y)==0:
                print(task_name,x,'num: 0')
            else:
                print(task_name,x,'num: {}, mean: {}, percentile 50: {}, percentile 80: {}, percentile 90: {}'.format(len(y),np.mean(y),round(np.percentile(y,50),2),round(np.percentile(y,80),2),round(np.percentile(y,90),2)))
        
        print('###################### topic2 ########################')
        for (x,y) in topic2.items():
            if x==None: continue
            if len(y)==0:
                print(task_name,x,'num: 0')
            else:
                print(task_name,x,'num: {}, mean: {}, percentile 50: {}, percentile 80: {}, percentile 90: {}'.format(len(y),np.mean(y),round(np.percentile(y,50),2),round(np.percentile(y,80),2),round(np.percentile(y,90),2)))
    except:
        import pdb 
        pdb.set_trace()
    import pdb 
    pdb.set_trace()


def parallel_func(input):
    (args,task_name)=input
    print('start processing task {}'.format(task_name))
    set_random_seed(args.seed)
    print(task_name)
    if task_name in ['glue/qqp','kilt_tasks/hotpotqa','wiki_qa','cos_e/v1.11','quarel','sciq','trec']:
        print('finished task {}'.format(task_name))
        return None
    already_save_path='../../huggingface_datasets/jing_crossed_dataset_unfiltered/aug_T0_v7_onlyparaqa/{}'.format(task_name)
    save_path='../../huggingface_datasets/jing_crossed_dataset_unfiltered/aug_T0_v8/{}'.format(task_name)
    if os.path.exists(os.path.join(save_path,'candidates_{}.npy'.format(args.fold_id)))==True:
        print('finished task {}'.format(task_name))
        return None
        tmp=np.load(os.path.join(save_path,'candidates_{}.npy'.format(args.fold_id)),allow_pickle=True).item()
    else:
        already_data=np.load(os.path.join(already_save_path,'candidates_{}.npy'.format(args.fold_id)),allow_pickle=True).item()
        tmp=get_prompted_candidates_onlyupdateqa(args,task_name,already_data=already_data)
    
    lengths=[(n,len(x)) for (n,x) in tmp.items()]
    print(lengths,sum([x[1] for x in lengths]))
    
    if os.path.exists(save_path)==False:
        os.makedirs(save_path)
    np.save(os.path.join(save_path,'candidates_{}.npy'.format(args.fold_id)),tmp)
    print('finished task {}'.format(task_name))
    return None

def build_global_cls_uniformed_dataset(input):
    (args,task_name,subtask_name)=input
    if task_name in ['glue/qqp','kilt_tasks/hotpotqa','wiki_qa','cos_e/v1.11','quarel','sciq','trec']: return []
    uniformed_dataset=build_uniformed_dataset_from_crossdatasets_0527(args,task_name)
    if subtask_name=='sentiment_2':
        if task_name in ['amazon_polarity','imdb','rotten_tomatoes','app_reviews','yelp_review_full']: return []
        data=filter_sentiment2_dataset(args,task_name,uniformed_dataset,th=1.0,with_prob=True)
    elif subtask_name=='sentiment_5':
        if task_name in ['amazon_polarity','imdb','rotten_tomatoes','app_reviews','yelp_review_full']: return []
        data=filter_sentiment5_dataset(args,task_name,uniformed_dataset,th=1.0,with_prob=True)
    elif subtask_name=='topic1':
        if task_name in ['ag_news']: return []
        data=filter_cls_dataset(args,task_name,uniformed_dataset,la_name=['attributes','topic1','answer'],th=1.0,with_prob=True)
    elif subtask_name=='topic2':
        if task_name in ['dbpedia_14']: return []
        data=filter_cls_dataset(args,task_name,uniformed_dataset,la_name=['attributes','topic2','answer'],th=1.0,with_prob=True)
    return data

def build_global_cls_dataset(args,task_names,subtask_name,parallel=True):
    assert subtask_name in ['sentiment_2','sentiment_5','topic1','topic2']
    total_data=[]
    if parallel==True:
        pool=Pool(processes=len(task_names))
        results=pool.map(build_global_cls_uniformed_dataset,[(args,task_name,subtask_name) for task_name in task_names])
        pool.close()
        total_data=results
    else:
        for task_name in task_names:
            print(task_name)
            data=build_global_cls_uniformed_dataset((args,task_name,subtask_name))
            total_data.append(data)
    # import pdb 
    # pdb.set_trace()
    if subtask_name=='sentiment_2':
        prompt_names=['paragraph_title_to_sentiment']
        prompts=sum([[y for (x,y) in uniformed_prompt_templates[prompt_name].items()] for prompt_name in prompt_names],[])
        prompts+=[uniformed_prompt_templates['paragraph_to_sentiment'][x] for x in ['User_recommend_this_product', 'Movie Expressed Sentiment 2', 'Reviewer Opinion bad good choices', 'Sentiment with choices ', 'Reviewer Sentiment Feeling', 'Writer Expressed Sentiment', 'Movie Expressed Sentiment', 'Text Expressed Sentiment', 'Negation template for positive and negative', 'Reviewer Enjoyment Yes No', 'Reviewer Expressed Sentiment','Reviewer Enjoyment']]
        
        sentiment2_data={0:[],1:[]};sentiment2_logits={0:[],1:[]}
        for data in total_data:
            tmp_sentiment2_data={0:[],1:[]};tmp_sentiment2_logits={0:[],1:[]}
            for e in data:
                new_e=copy.deepcopy(e)
                new_e['attributes']['sentiment_2']['answer']=e['attributes']['sentiment_2']['answer'][0]
                tmp_sentiment2_data[e['attributes']['sentiment_2']['answer'][0]].append(new_e)
                tmp_sentiment2_logits[e['attributes']['sentiment_2']['answer'][0]].append(e['attributes']['sentiment_2']['answer'][1])
            if len(tmp_sentiment2_data[0])<len(tmp_sentiment2_data[1])*10 and len(tmp_sentiment2_data[1])<len(tmp_sentiment2_data[0])*10:
                # if the times of data of two datasets exceeds 10, this dataset might not be suitable for sentiment analysis
                sentiment2_data[0]+=tmp_sentiment2_data[0];sentiment2_data[1]+=tmp_sentiment2_data[1]
                sentiment2_logits[0]+=tmp_sentiment2_logits[0];sentiment2_logits[1]+=tmp_sentiment2_logits[1]

        examples=[]
        print(0,np.percentile(sentiment2_logits[0],90));print(1,np.percentile(sentiment2_logits[1],90))
        slen1=len(sentiment2_data[0]);slen2=len(sentiment2_data[1]);th=0.2
        top_num=int(min(slen1,slen2)*th)
        print('select {} examples'.format(top_num))
        for (key,values) in sentiment2_logits.items():
            sorted_idxs=np.argsort(-np.array(values))
            slen=len(values)
            examples+=[sentiment2_data[key][sorted_idxs[i]] for i in range(top_num)]
        # prompted_datas=apply_prompts(examples,prompts)
    elif subtask_name=='sentiment_5':
        prompt_names=['sentiment_title_to_paragraph']
        prompts=sum([[y for (x,y) in uniformed_prompt_templates[prompt_name].items()] for prompt_name in prompt_names],[])
        prompts+=[uniformed_prompt_templates['paragraph_to_sentiment'][x] for x in ['categorize_rating_using_review', 'convert_to_star_rating', 'convert_to_rating', 'so_i_would', 'based_on_that', 'format_star', 'this_place', 'format_score', 'on_a_scale', 'format_rating']]
        
        sentiment_data={1:[],2:[],3:[],4:[],5:[]}; sentiment_logits={1:[],2:[],3:[],4:[],5:[]}
        for data in total_data:
            tmp_sentiment_data={1:[],2:[],3:[],4:[],5:[]}; tmp_sentiment_logits={1:[],2:[],3:[],4:[],5:[]}
            for e in data:
                new_e=copy.deepcopy(e)
                new_e['attributes']['sentiment_5']['answer']=e['attributes']['sentiment_5']['answer'][0]
                tmp_sentiment_data[e['attributes']['sentiment_5']['answer'][0]].append(new_e)
                tmp_sentiment_logits[e['attributes']['sentiment_5']['answer'][0]].append(e['attributes']['sentiment_5']['answer'][1])
            if len(tmp_sentiment_data[1])+len(tmp_sentiment_data[2])<(len(tmp_sentiment_data[4])+len(tmp_sentiment_data[5]))*10 and \
                len(tmp_sentiment_data[4])+len(tmp_sentiment_data[5])<(len(tmp_sentiment_data[1])+len(tmp_sentiment_data[2]))*10:
                # if the times of data of two datasets exceeds 10, this dataset might not be suitable for sentiment analysis
                for i in range(1,6):
                    sentiment_data[i]+=tmp_sentiment_data[i]
                    sentiment_logits[i]+=tmp_sentiment_logits[i]
        examples=[]
        top_num=100000000
        for key in range(1,6): 
            print(key,np.percentile(sentiment_logits[key],90))
            top_num=min(top_num,len(sentiment_logits[key]))
        th=0.2
        for (key,values) in sentiment_logits.items():
            sorted_idxs=np.argsort(-np.array(values))
            slen=len(values)
            examples+=[sentiment_data[key][sorted_idxs[i]] for i in range(int(top_num*th))]
        # import pdb 
        # pdb.set_trace()      
    elif subtask_name=='topic1':
        # {'company': [], 'educational institution': [], 'artist': [], 'athlete': [], 'office holder': [], 'mean of transportation': [], 'building': [], 'natural place': [], 'village': [], 'animal': [], 'plant': [], 'album': [], 'film': [], 'written work': []}
        prompt_names=['paragraph_to_topic1']
        prompts=sum([[y for (x,y) in uniformed_prompt_templates[prompt_name].items()] for prompt_name in prompt_names],[])
        
        topic_data={'World politics': [], 'Business': [], 'Science and technology': [], 'Sports': []}
        topic_logits={'World politics': [], 'Business': [], 'Science and technology': [], 'Sports': []}
        for data in total_data:
            tmp_topic_data={'World politics': [], 'Business': [], 'Science and technology': [], 'Sports': []}
            tmp_topic_logits={'World politics': [], 'Business': [], 'Science and technology': [], 'Sports': []}
            for e in data:
                new_e=copy.deepcopy(e)
                new_e['attributes']['topic1']['answer']=e['attributes']['topic1']['answer'][0]
                tmp_topic_data[e['attributes']['topic1']['answer'][0]].append(new_e)
                tmp_topic_logits[e['attributes']['topic1']['answer'][0]].append(e['attributes']['topic1']['answer'][1])
            for key in topic_data.keys():
                topic_data[key]+=tmp_topic_data[key]
                topic_logits[key]+=tmp_topic_logits[key]
        examples=[]
        top_num=100000000
        for key in topic_data.keys(): 
            print(key,np.percentile(topic_logits[key],90))
            top_num=min(top_num,len(topic_logits[key]))
        th=0.2
        top_num=int(top_num*th)
        for (key,values) in topic_logits.items():
            sorted_idxs=np.argsort(-np.array(values))
            slen=len(values)
            examples+=[topic_data[key][sorted_idxs[i]] for i in range(top_num)]
            # import pdb 
            # pdb.set_trace()
        # prompted_datas=apply_prompts(examples,prompts)
    elif subtask_name=='topic2':
        prompt_names=['paragraph_to_topic2','title_to_topic2','paragraph_title_to_topic2']
        prompts=sum([[y for (x,y) in uniformed_prompt_templates[prompt_name].items()] for prompt_name in prompt_names],[])
        
        topic_data={'company': [], 'educational institution': [], 'artist': [], 'athlete': [], 'office holder': [], 'mean of transportation': [], 'building': [], 'natural place': [], 'village': [], 'animal': [], 'plant': [], 'album': [], 'film': [], 'written work': []}
        topic_logits={'company': [], 'educational institution': [], 'artist': [], 'athlete': [], 'office holder': [], 'mean of transportation': [], 'building': [], 'natural place': [], 'village': [], 'animal': [], 'plant': [], 'album': [], 'film': [], 'written work': []}
        for data in total_data:
            tmp_topic_data={'company': [], 'educational institution': [], 'artist': [], 'athlete': [], 'office holder': [], 'mean of transportation': [], 'building': [], 'natural place': [], 'village': [], 'animal': [], 'plant': [], 'album': [], 'film': [], 'written work': []}
            tmp_topic_logits={'company': [], 'educational institution': [], 'artist': [], 'athlete': [], 'office holder': [], 'mean of transportation': [], 'building': [], 'natural place': [], 'village': [], 'animal': [], 'plant': [], 'album': [], 'film': [], 'written work': []}
            for e in data:
                new_e=copy.deepcopy(e)
                new_e['attributes']['topic2']['answer']=e['attributes']['topic2']['answer'][0]
                tmp_topic_data[e['attributes']['topic2']['answer'][0]].append(new_e)
                tmp_topic_logits[e['attributes']['topic2']['answer'][0]].append(e['attributes']['topic2']['answer'][1])
            for key in topic_data.keys():
                topic_data[key]+=tmp_topic_data[key]
                topic_logits[key]+=tmp_topic_logits[key]
        examples=[]
        top_num=100000000
        for key in topic_data.keys(): 
            print(key,len(topic_logits[key]))
            print(key,np.percentile(topic_logits[key],90))
            top_num=min(top_num,len(topic_logits[key]))
        th=0.2
        top_num=int(top_num*th)
        for (key,values) in topic_logits.items():
            sorted_idxs=np.argsort(-np.array(values))
            slen=len(values)
            examples+=[topic_data[key][sorted_idxs[i]] for i in range(top_num)]
            # import pdb 
            # pdb.set_trace()
    if parallel==True and len(examples)>5000:
        block_num=100
        blocked_examples=[]
        for i in range(int(len(examples)//block_num)):
            st=i*block_num;ed=(i+1)*block_num
            blocked_examples.append(examples[st:ed])
        blocked_examples.append(examples[ed:])
        pool=Pool(processes=block_num)
        results=[]
        for es in blocked_examples:
            ans=pool.apply_async(apply_prompts,(es,prompts))
            results.append(ans)
        pool.close()
        pool.join()
        results=[res.get() for res in results]
        prompted_datas=sum(results,[])
    else:
        prompted_datas=apply_prompts(examples,prompts)
        
    # import pdb 
    # pdb.set_trace()
    return prompted_datas

# task_names="glue/mrpc paws/labeled_final adversarial_qa/dbidaf amazon_polarity app_reviews imdb rotten_tomatoes yelp_review_full common_gen cnn_dailymail/3.0.0 gigaword multi_news samsum xsum ag_news dbpedia_14 adversarial_qa/dbert adversarial_qa/droberta duorc/SelfRC duorc/ParaphraseRC ropes quoref"
# for task_name in task_names:
#     data=np.load(os.path.join(save_path,'task_name','candidates_{}.npy'.format(np.fold_id)).allow_pickle=True).item()
#     data['qa_tf'],data['qa_extractive'],data['qa_multi_choice']
if __name__ == '__main__':
    from universal_da.dataset_config import default_T0_tasks
    from SwissArmyTransformer.training.deepspeed_training import initialize_distributed, set_random_seed
    # from arguments import get_args
    # args,_=get_args()
    import argparse
    from arguments import get_args
    py_parser = argparse.ArgumentParser(add_help=False)
    T5Model.add_model_specific_args(py_parser)
    known, args_list = py_parser.parse_known_args()
    args = get_args(args_list)
    args = argparse.Namespace(**vars(args), **vars(known))
    torch.backends.cudnn.enabled = False

    uniformed_datasets=dict()
    dataset=dict()
    prompted_datas=dict()
    task_names=sum([names for (_,names) in default_T0_tasks.items()],[])
    # ##################################################################################################################
    # selected_data=build_global_cls_dataset(args,task_names,'sentiment_2')
    # save_path='../../huggingface_datasets/jing_crossed_dataset_unfiltered/aug_T0_v8/sentiment2'
    # selected_data=build_global_cls_dataset(args,task_names,'sentiment_5')
    # save_path='../../huggingface_datasets/jing_crossed_dataset_unfiltered/aug_T0_v8/sentiment5'
    # selected_data=build_global_cls_dataset(args,task_names,'topic1')
    # save_path='../../huggingface_datasets/jing_crossed_dataset_unfiltered/aug_T0_v8/topic1'
    # selected_data=build_global_cls_dataset(args,task_names,'topic2')
    # save_path='../../huggingface_datasets/jing_crossed_dataset_unfiltered/aug_T0_v8/topic2'

    selected_data=build_global_cls_dataset(args,task_names,'sentiment_2')
    save_path='../../huggingface_datasets/jing_crossed_dataset_unfiltered/aug_T0_v8/sentiment2_small'
    # selected_data=build_global_cls_dataset(args,task_names,'sentiment_5')
    # save_path='../../huggingface_datasets/jing_crossed_dataset_unfiltered/aug_T0_v8/sentiment5_small'
    # selected_data=build_global_cls_dataset(args,task_names,'topic1')
    # save_path='../../huggingface_datasets/jing_crossed_dataset_unfiltered/aug_T0_v8/topic1_small'
    # selected_data=build_global_cls_dataset(args,task_names,'topic2')
    # save_path='../../huggingface_datasets/jing_crossed_dataset_unfiltered/aug_T0_v8/topic2_small'
    if os.path.exists(save_path)==False:
        os.makedirs(save_path)
    np.save(os.path.join(save_path,'candidates_{}.npy'.format(args.fold_id)),selected_data)
    # ##################################################################################################################
    # for task_name in task_names:
    #     if task_name in ['glue/qqp','kilt_tasks/hotpotqa','wiki_qa','cos_e/v1.11','quarel','sciq','trec']:
    #         continue
    #     if task_name !='gigaword': continue
    #     parallel_func((args,task_name))
    #     uniformed_dataset=build_uniformed_dataset_from_crossdatasets_0527(args,task_name)
    #     test_cls_tasks(uniformed_dataset,task_name)
    #     dataset['qa']=filter_qa_dataset(args,task_name,uniformed_dataset,th=1.0)
    #     dataset['paraphrase']=filter_paraphrase_dataset(args,task_name,uniformed_dataset)
    #     dataset['sentiment_2']=filter_cls_dataset(args,task_name,uniformed_dataset,la_name=['attributes','sentiment_2','answer'])
    #     dataset['sentiment_5']=filter_cls_dataset(args,task_name,uniformed_dataset,la_name=['attributes','sentiment_5','answer'])
    #     dataset['topic1']=filter_cls_dataset(args,task_name,uniformed_dataset,la_name=['attributes','topic1','answer'])
    #     dataset['topic2']=filter_cls_dataset(args,task_name,uniformed_dataset,la_name=['attributes','topic2','answer'])
    #     dataset['keywords']=filter_gen_dataset(args,task_name,uniformed_dataset,th=1.0,la_name=['attributes','keywords','answer'])
    #     dataset['title']=filter_gen_dataset(args,task_name,uniformed_dataset,th=1.0,la_name=['attributes','title','answer'])
    #     import pdb 
    #     pdb.set_trace()
    # ########################################################################################################
    # pool=Pool(processes=38)
    # results=pool.map(parallel_func,[(args,task_name) for task_name in task_names])
    # pool.close()
    # ########################################################################################################
    # for (i,task_name) in enumerate(task_names):
    #     set_random_seed(args.seed)
    #     print(task_name)
    #     if task_name in ['glue/qqp','kilt_tasks/hotpotqa','wiki_qa','cos_e/v1.11','quarel','sciq','trec']:
    #         continue
    #     already_save_path='../../huggingface_datasets/jing_crossed_dataset_unfiltered/aug_T0_v4/{}'.format(task_name)
    #     save_path='../../huggingface_datasets/jing_crossed_dataset_unfiltered/aug_T0_v6/{}'.format(task_name)
    #     if os.path.exists(os.path.join(save_path,'candidates_{}.npy'.format(args.fold_id)))==True:
    #         continue
    #         tmp=np.load(os.path.join(save_path,'candidates_{}.npy'.format(args.fold_id)),allow_pickle=True).item()
    #     else:
    #         # try:
    #         already_data=np.load(os.path.join(already_save_path,'candidates_{}.npy'.format(args.fold_id)),allow_pickle=True).item()
    #         tmp=get_prompted_candidates_onlyupdateqa(args,task_name,already_data=already_data)
    #         # except:
    #         #     import pdb 
    #         #     pdb.set_trace()
    #     # tmp=get_prompted_candidates(args,task_name)
    #     # prompted_datas[task_name]=tmp
        
    #     lengths=[(n,len(x)) for (n,x) in tmp.items()]
    #     print(lengths,sum([x[1] for x in lengths]))
        
    #     if os.path.exists(save_path)==False:
    #         os.makedirs(save_path)
    #     np.save(os.path.join(save_path,'candidates_{}.npy'.format(args.fold_id)),tmp)





'''
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
--fold_id 0
'''
    