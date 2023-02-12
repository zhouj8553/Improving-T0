import os
import logging
import random
import copy
import argparse
import time
import numpy as np

from universal_da.T0_data_utils import *
from universal_da.simple_cross import convert_dataset_uniform,uniform_prompts
from universal_da.dataset_config import default_T0_tasks


prompt_map=["para1","para2","para1_meta","{{question}}","questions_meta","{{answer}}","answer_label","{{choices",
'attributes["title"]["answer"]','attributes["sentiment_2"]["answer"]','attributes["sentiment_5"]["answer"]','attributes["keywords"]["answer"]',
'attributes["topic1"]["answer"]','attributes["topic1"]["candidates"]','attributes["topic2"]["answer"]','attributes["topic2"]["candidates"]']

example_map=[
    "para1","para2",["para1_meta","similar_sentence"],"question",["questions_meta","similar_sentence"],"answer","answer_label","choices",["attributes","title","answer"],["attributes","sentiment_2","answer"],["attributes","sentiment_5","answer"],["attributes","keywords","answer"],
    ["attributes","topic1","answer"],["attributes","topic1","candidates"],["attributes","topic2","answer"],["attributes","topic2","candidates"]
]
def prepare_prompt_codes(prompts):
    all_match_value=int(''.join(['1' for _ in range(len(prompt_map))]),2)
    def get_prompt_code(prompt):
        prompt_descriptions,prompt_target=prompt.__dict__['jinja'].split('|||')
        # use binary
        ans1='';ans2=''
        for name in prompt_map:
            tmp1='0' if name in prompt_descriptions else '1'
            tmp2='0' if name in prompt_target else '1' 
            if name=='{{choices' and prompt.answer_choices is not None and ('choices' in prompt.answer_choices or 'A' in prompt.answer_choices):
                if 'choices' in prompt_descriptions: tmp1='0';
                if 'choices' in prompt_target: tmp2='0'
            ans1+=tmp1;ans2+=tmp2
        return (ans1,ans2)
    prompt_codes=[]
    for prompt in prompts:
        prompt_codes.append(get_prompt_code(prompt))
    return prompt_codes,all_match_value

def prepare_example_codes(examples):
    def get_example_code(e):
        ans=''
        for name in example_map:
            if isinstance(name,list):
                if len(name)==3:
                    # print(e,name)
                    if name[0] in e and e[name[0]][name[1]][name[2]]!=None and e[name[0]][name[1]][name[2]]!=[]: tmp='1'
                    else: tmp='0'
                elif len(name)==2:
                    if name[0] in e and e[name[0]][name[1]]!=None and e[name[0]][name[1]]!=[]: tmp='1'
                    else: tmp='0'
                else: raise NotImplementedError
            else:
                if name=='choices':
                    if name in e and e[name]!=None and (isinstance(e[name],list)==False or len(e[name])>1): tmp='1'
                    else: tmp='0'
                else:
                    if name in e and e[name]!=None and (isinstance(e[name],list)==False or len(e[name])>0): tmp='1'
                    else: tmp='0'
            ans+=tmp
        return ans
    example_codes=[]
    for e in examples:
        example_codes.append(get_example_code(e))
    return example_codes

