import numpy as np
import os
from data_utils import *
all_tasks_ls = T0_TRAIN_TASK_LIST
all_tasks_ls = [t.replace("/","_") for t in all_tasks_ls]
root_file = "/share/zongyu/t0_withemb"
for i, (root,dirs,file) in enumerate(os.walk(root_file)):
    if i == 0:
        continue
    if "PAQ" in root:
        continue
    tmp_task = root.split("/")[-1]
    flag = False
    for cand in all_tasks_ls: # all_tasks_ls
        if cand in tmp_task:
            flag = True
            break
    if not flag:
        continue
