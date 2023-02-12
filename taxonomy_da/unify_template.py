# 统一template字段，保证每一个分支里的都一样
from collections import defaultdict
import os

from promptsource.templates import DatasetTemplates
import string
import pandas as pd
import numpy as np
from tool import *
from data_utils import T0_TRAIN_TASK_LIST,T0_TEST_TASK_LIST, TASK_TYPE_DICT
from datasets import concatenate_datasets, load_dataset,Dataset,load_from_disk
import json
import torch
from kfold.tasks.p3.p3 import P3_TASK_LIST
from tqdm import tqdm
# import random
# from accelerate import Accelerator
from classification_tree import *
from init_func import search_tree, pair2func, _sample_train_data, tpn2branch_revise, \
    rep_map, revise_raw_data, print_test
from collections import defaultdict
import random
# random.seed(42)
rng = random.Random(42)

preserve_col = ["inputs_pretokenized","targets_pretokenized"]

def unify_tpn(tpn):
    key_ls = string.punctuation + " "
    trantab = str.maketrans({key: "_" for key in key_ls})
    tpn = tpn.translate(trantab)
    return tpn
def apply_prompt(x,prompt):
    results = prompt.apply(x)
    if len(results) == 1: # None
        x["inputs_pretokenized"] = None
        x["targets_pretokenized"] = None
        return x
    else:
        x["inputs_pretokenized"] = results[0]
        x["targets_pretokenized"] = results[1]
        return x
def remove_template(example, jinja):
    return 0
def write_file(dataset, data_dir=None):
    write_file = open(data_dir, "w")
    for item in dataset:
        line=json.dumps(item)
        write_file.write(line + "\n")
    write_file.close()

seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)

train_ls = T0_TRAIN_TASK_LIST
# train_ls = [t.replace("/","_") for t in train_ls]
raw_data_dir = "/share/zongyu/zhoujing/huggingface_datasets"
save_dir = "/share/zongyu/data/t0_all_create"


match_dict = torch.load("/share/zongyu/task_embedding/ours/output/prompt_revise_dict.pt")
match_dict = {v:i for i,v in match_dict.items()}
# 加一下match_dict


finish = True
# 做一下reverse的步骤
tpn2pn = defaultdict(tuple) # template中的promptname -> 真实文件名tpn的映射
pn2pn = {} # template中的pn到真实的pn
# print_test()
for tn in train_ls:
    prompts = DatasetTemplates(tn) # revise
    all_prompt_names = prompts.all_template_names
    for index, prompt_name in enumerate(all_prompt_names):
        # print("The prompt_name is {}.".format(prompt_name))
        prompt = prompts[prompt_name]
        test_tpn = tn + "_" + prompt_name
        test_tpn = unify_tpn(test_tpn)
        test_tpn = match_dict[test_tpn]
        tpn2pn[test_tpn] = (tn,prompt_name)
        real_pn = test_tpn[len(tn)+1:]
        pn2pn[prompt_name] = real_pn

P3_train_ls = []
for tn in train_ls:
    tn = tn.replace("/","_")
    cand_ls = [p for p in P3_TASK_LIST if p.startswith(tn)]
    P3_train_ls += cand_ls

data_dir = "/share/zongyu/t0_cross_all"

# for tpn in P3_train_ls:
#     tn, pn = tpn2pn[tpn]
#     read_data_path = os.path.join(data_dir,tpn,"train.json")
#     data = load_dataset("json", path=read_data_path, cache_dir="/root/.cache/hugginface/datasets")
#     data = data['train']
    
# 每个分支都按照那个template来修正jinja
debug = False
tax2tpns = {}
for c1,t1 in clf_tree_fine.items():
    for c2,t2 in t1.items():
        for c3,t3 in t2.items():
            for c4,tpns in t3.items():
                tax = "_".join([c1,c2,c3,c4])
                tax2tpns[tax] = tpns
tax_ls = list(tax2tpns.keys())
span1 = list(range(0,4))
span2 = list(range(4,7))
span3 = list(range(7,14))
span4 = list(range(14,16))
spanls = [span1,span2,span3,span4]
spanid = 2
span = spanls[spanid]
write = True
write_to_disk = True
# for c1,t1 in clf_tree_fine.items():
#     for c2,t2 in t1.items():
#         for c3,t3 in t2.items():
#             for c4,tpns in t3.items():
#                 tax = "_".join([c1,c2,c3,c4]) # record 当前分支
bad_test_ls = ['wiqa','ropes','qasc','common_gen','kilt_tasks/hotpotqa','adversarial_qa/dbidaf','adversarial_qa/dbert','adversarial_qa/droberta']
save_dir = "/share/zongyu/data/huggingface_datasets_0416"
for cnt, (tax, tpns) in enumerate(tax2tpns.items()):
    if cnt not in span:
        continue
    # if tax == "sin_sent_disc_fix_ls_parel_1": # summarization extension drop
    #     continue
    # if tax != "double_sent_gen_create_ccs_1": # DEBUG
    #     continue
    print(f"Current branch:{tax}")
    tns = []
    tn2func = {}
    for j, tpn in enumerate(tpns):
        tn, pn = tpn2pn[tpn]
        if tpn not in tpn2branch_revise:
            continue
        branch, cur_dict = tpn2branch_revise[tpn]
        tns.append(tn)
        tn2func[tn] = cur_dict['data']
    # 0317 add sub tasks
    if "adversarial_qa/dbidaf" in tns:
        tns.append("adversarial_qa/dbert")
        tns.append("adversarial_qa/droberta")
        tn2func["adversarial_qa/dbert"] = tn2func["adversarial_qa/dbidaf"]
        tn2func["adversarial_qa/droberta"] = tn2func["adversarial_qa/dbidaf"]
    if "imdb" in tns:
        tns.append("rotten_tomatoes")
        tn2func["rotten_tomatoes"] = tn2func["imdb"]
    if "duorc/SelfRC" in tns:
        tns.append("duorc/ParaphraseRC")
        tn2func["duorc/ParaphraseRC"] = tn2func["duorc/SelfRC"]
    tns = list(set(tns))
    for tn in tns:
        print(f"Now process tn:{tn}")
        # del
        # if "ag_news" not in tn:
        #     continue
        #
        read_data_path = os.path.join(raw_data_dir,tn)
        if os.path.exists(read_data_path):
            data = load_from_disk(read_data_path)
        else:
            print(f"ERROR!{tn} do not exists!")
        # data = data['train']
        # data = data.select(range(10))
        map_data_func = tn2func[tn]
        if len(list(data.keys())) == 1:
            data = data['train'].train_test_split()
        
        for split, cur_data in data.items():
            print(f"Now Split:{split}")
            # cur_data = cur_data.shuffle(seed=42).select(list(range(100))) # debug
            # if split != "train": # debug
            #     continue
            if tn in bad_test_ls: # no answeror label
                if split == "test":
                    continue
            cur_data = cur_data.map(map_data_func, num_proc=32,load_from_cache_file=False)
            for i, tpn in enumerate(tpns):
                # if "wiki_qa_Generate_Question_from_Topic" not in tpn: # debug
                #     continue
                # cur_data_ = _sample_train_data(tn,cur_data,rng)
                cur_data_ = cur_data # 0416
                # print(cur_data_[1])
                print(f"Now transfer {tn} to tpn:{tpn}")
                print("#"*100)
                tnn, pn = tpn2pn[tpn]
                if tnn == tn:
                    continue
                if "adversarial_qa" in tnn and "adversarial_qa" in tn:
                    continue
                if "duorc" in tnn and "duorc" in tn:
                    continue
                if "imdb" in tnn and "rotten" in tn:
                    continue
                if "rotten" in tnn and "imdb" in tn:
                    continue
                template = DatasetTemplates(tnn)[pn]
                print(template.jinja)
                print("#"*100)
                if tpn not in tpn2branch_revise:
                    print("Not done or cannot use!")
                    continue
                branch, cur_dict = tpn2branch_revise[tpn]
                new_jinja = rep_map(template, cur_dict['jinja'])
                print("new_jinja")
                print("#"*100)
                print(new_jinja)
                print("#"*100)
                template.jinja = new_jinja

                # map_data_func = cur_dict['data']
                prompted_data = cur_data_.map(lambda x:apply_prompt(x,template), num_proc=32,load_from_cache_file=False)

                prompted_data = prompted_data.filter(lambda x: x['inputs_pretokenized'])
                prompted_data = prompted_data.filter(lambda x: x['targets_pretokenized'])
                all_cols = prompted_data.column_names
                rm_cols = [c for c in all_cols if c not in preserve_col]
                prompted_data = prompted_data.remove_columns(rm_cols)
                if len(prompted_data) > 0 and len(prompted_data[0]) > 0: # len_key > 0
                    if not prompted_data[0]['inputs_pretokenized'] or not prompted_data[0]['targets_pretokenized']:
                        continue
                    else:
                        print(prompted_data[0])
                        print("#"*100)
                else:
                    print("Created data is NULL!")
                    continue
                print(f"finish unify task:{tn} to branch:{branch}")
                new_tn_tpn = tn.replace("/","_") + "-" + tpn
                save_path = os.path.join(save_dir,new_tn_tpn)
                print(f"Begin save file:{save_path}, data size{len(prompted_data)}")
                print(f"If exists:{os.path.exists(save_path)}")
                # save
                if write:
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                        if write_to_disk:
                            save_file_path = save_path + "/" + split
                            prompted_data.save_to_disk(save_file_path)
                        else:
                            save_file_path = save_path + "/" + split + ".json"
                            write_file(prompted_data,save_file_path)
                    else:
                        if write_to_disk:
                            save_file_path = save_path + "/" + split
                            prompted_data.save_to_disk(save_file_path)
                        else:
                            save_file_path = save_path + "/" + split + ".json"
                            write_file(prompted_data,save_file_path)
                        print("Already exists!")

