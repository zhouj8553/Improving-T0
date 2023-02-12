# 直接按照task family分成kfold
# 先只验证parapharase identification
# TODO: 写成多个kfold的形式，可以通过手动修改的形式跑两轮来解决

import os
import torch
from data_utils import T0_TRAIN_TASK_LIST, TASK_TYPE_DICT
from datasets import concatenate_datasets, load_dataset,Dataset
import json
import pandas as pd
import numpy as np
from tasks.p3.p3 import P3_TASK_LIST
from tool import *

DEBUG = True
mode = "domain_task" # 选取什么样的embedding
output_path = f"data/selected_t0_domain_task_wic"
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


t0_train_ls = T0_TRAIN_TASK_LIST
taskfamily2task = TASK_TYPE_DICT
family_set = list(taskfamily2task.keys())
task2family = {}
for task_family, tasks in taskfamily2task.items():
    for task in tasks:
        if task in t0_train_ls:
            task2family[task.replace("/","_")] = task_family

holdout_task_family = ["paraphrase_identification"]
k_1_taskfamily = [tf for tf in family_set if tf not in holdout_task_family]
k_1_tasks = []
holdout_tasks = []
for tf1 in k_1_taskfamily:
    k_1_tasks += taskfamily2task[tf1]

for tf2 in holdout_task_family:
    holdout_tasks += taskfamily2task[tf2]

k_1_tasks = [t.replace("/","_") for t in k_1_tasks]
holdout_tasks = [t.replace("/","_") for t in holdout_tasks]

# 根据holdout task去k-1份里面去retrieve
def get_data(data_files, extension="json"):
    raw_datasets = load_dataset(extension, data_files=data_files,cache_dir="/share/zongyu/cache/huggingface/datasets") #  cache_dir="/share/zongyu/cache/huggingface/datasets"
    return raw_datasets
def get_emb(data_files):
    embs = torch.load(data_files)
    return embs
def write_file(dataset, data_dir=None):
    write_file = open(data_dir, "w")
    for item in dataset:
        line=json.dumps(item)
        write_file.write(line + "\n")
    write_file.close()
# Retrieve的超参数
topk_ = 2048
root_path = "/share/zongyu/t0_emb"
# Read query embedding first

# 首先生成query的文件列表（embedding文件）
query_file_ls = []
query_emb_ls = []
for task_name in holdout_tasks:
    task_name = task_name.replace("/", "_")
    sub_list = [task_li for task_li in P3_TASK_LIST if task_li.startswith(task_name)]
    query_file_ls = query_file_ls + sub_list # TODO
query_file_ls = [os.path.join(root_path,query,"train.json") for query in query_file_ls]

for i, query_file in enumerate(query_file_ls):
    # read_path = os.path.join(root_file)
    if DEBUG:
        if i > 1:
            break
    query_ds = get_data(query_file)
    query_ds = query_ds['train']
    print(f"Begin processing:{query_file}")
    query_sents = [d['inputs_pretokenized'] for d in query_ds]
    query_file_emb = query_file.replace(".json",f"_emb_{mode}.pt")
    query_emb = torch.load(query_file_emb)
    query_emb = query_emb / np.linalg.norm(query_emb,axis=1,keepdims=True)
    query_emb_ls.append(query_emb)

query_emb_ls = np.concatenate(query_emb_ls)


# 读取全部的data，我们首先生成要读取的文件的列表
cand_file_ls = []

sent_dict = {}
embs_dict = {}
globalidx2localdata = [] # 记录全局index到局部数据集的dict
globalidx2localidx = [] # 记录全局index到局部数据集index的dict
for task_name in k_1_tasks:
    task_name = task_name.replace("/", "_")
    sub_list = [task_li for task_li in P3_TASK_LIST if task_li.startswith(task_name)]
    cand_file_ls = cand_file_ls + sub_list

cand_file_ls = [os.path.join(root_path,cand,"train.json") for cand in cand_file_ls]
global_cnt = 0
for i, cand_file in enumerate(cand_file_ls):
    if DEBUG:
        if i > 5:
            break
    cand_ds = get_data(cand_file)
    cand_ds = cand_ds['train']
    print(f"Begin processing:{cand_file}")
    cand_sents = [d['inputs_pretokenized'] for d in cand_ds]
    cand_file_emb = cand_file.replace(".json",f"_emb_{mode}.pt")
    simcse_embs = torch.load(cand_file_emb)
    simcse_embs = simcse_embs.astype(np.float32)
    simcse_embs = simcse_embs / np.linalg.norm(simcse_embs,axis=1,keepdims=True)
    embs_dict[cand_file_emb] = simcse_embs
    sent_dict[cand_file] = []
    for local_idx, d in enumerate(cand_ds):
        # sent_dict[read_path].append(d['inputs_pretokenized']+d['targets_pretokenized'])
        sent_dict[cand_file].append(d['inputs_pretokenized'])
        globalidx2localdata.append(cand_file.replace("t0_emb","preprocessed_t0"))
        globalidx2localidx.append(local_idx)
        # global_cnt += 1


# 最后只保存index
# Output: {"task_prompt_name_path_1":[idx1,idx2,...],...}
model_name = "/share/zongyu/huggingface_models/bert-base-uncased"
simcse = SimCSE(model_name)
simcse.build_index_emb(sent_dict,embs_dict,use_faiss=True, faiss_compress=False,device=device)
# results, chosen_idxs, topk_sents = simcse.search_emb(selected_anchor_embs, test_sents,threshold=0.6,top_k=2048) # explore topk
 
results, chosen_idxs = simcse.search_emb(query_emb_ls, query_sents,threshold=0.6,top_k=2048)


def save_index(chosen_idxs_,globalidx2localdata_,globalidx2localidx_):
    output_dict = {}
    ori_len = len(globalidx2localdata_)
    data_path_set = list(set(globalidx2localdata_))
    globalidx2localdata_ = [globalidx2localdata_[cid] for cid in chosen_idxs_]
    globalidx2localidx_ = [globalidx2localidx_[cid] for cid in chosen_idxs_]
    print(f"Select {len(globalidx2localdata_)} from {ori_len}")
    for task_prompt_name in data_path_set:
        cur_task_idxs = [globalidx2localidx_[gid] for gid,d in enumerate(globalidx2localdata_) if d==task_prompt_name]
        # cur_task_idxs = globalidx2localidx_[globalidx2localdata_==task_prompt_name]
        output_dict[task_prompt_name] = cur_task_idxs
    holdout = holdout_task_family[0]
    torch.save(output_dict,f"/share/zongyu/task_embedding/ours/kfold/output/idxs_{mode}_{holdout}")
    return output_dict

def load_index(index_path):
    output_dict = torch.load(index_path)
    return output_dict
output_dict = save_index(chosen_idxs,globalidx2localdata,globalidx2localidx)
# check 
holdout = holdout_task_family[0]
output_dict_ = load_index(f"/share/zongyu/task_embedding/ours/kfold/output/idxs_{mode}_{holdout}")