#!/usr/bin/env python
# coding=utf-8
# Copyright BigScience, The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Reproduce the main evaluation in `Multitask Prompted Training Enables Zero-Shot Task Generalization` using PyTorch.
This script is heavily adapted from https://github.com/huggingface/transformers/blob/7533d30acd975027e83a548e4c38e06fa335291b/examples/pytorch/multiple-choice/run_swag_no_trainer.py
"""

import argparse
import logging
import os
import random
import json
import shutil

import datasets
import numpy as np
import torch
# from datasets import load_dataset, load_metric
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    default_data_collator,
)
from promptsource.templates import DatasetTemplates

# from .move_t5_inverse import inverse_transform_ckpt
# from .t0.data_collator import DataCollatorForMultipleChoice
# from .t0.model import ModelBase
# from datasets import load_from_disk, load_metric
# from .template_list import template_list

from move_t5_inverse import inverse_transform_ckpt
from t0.data_collator import DataCollatorForMultipleChoice
from t0.model import ModelBase
from datasets import load_from_disk, load_metric
from train_template_list import template_list

logger = logging.getLogger(__name__)

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

def parse_args():
    parser = argparse.ArgumentParser(description="Reproduce main evaluation in T0.")

    parser.add_argument("--eval_data_dir", type=str, help="The directory of evaluation dataset.")
    parser.add_argument('--eval_model_dir', type=str)

    """
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
        required=True,
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--template_name",
        type=str,
        default=None,
        help="The template/prompt name",
        required=True,
    )
    """
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--target_max_length",
        type=int,
        default=256,
        help="Target max length. Sequences longer than this will be truncated."
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models. The list of T0 variants can be found on `https://huggingface.co/bigscience/T0_3B`",
        required=True,
    )
    """
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    """
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Where to store the final model."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Activate debug mode and run training only with a subset of data.",
    )
    parser.add_argument(
        "--reevaluate",
        action="store_true",
        help="Activate debug mode and run training only with a subset of data.",
    )

    parser.add_argument(
        "--parallelize",
        action="store_true",
        help=(
            "If passed, will call `model.parallelize` which splits the model on all GPUs available when applicable (model parallelism). "
            "Note that this feature is still experimental in HF Transformers."
        ),
    )
    parser.add_argument(
        "--local_rank",
        default=0,
    )
    parser.add_argument(
        "--max_valid_num",
        default=1000,
    )
    args = parser.parse_args()

    return args

def rearange_results(results):
    total_results={}
    for result in results:
        for (metric,value) in result.items():
            if metric in total_results:
                total_results[metric].append(value)
            else:
                total_results[metric]=[value]
    print(total_results)
    for metric,values in total_results.items():
        total_results[metric]=np.mean(values)
    return total_results

def compute(args, model, tokenizer, dataset_name, dataset_config_name, template_name, accelerator):

    # In distributed evaluation, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    """
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        if args.dataset_name == "anli":
            raw_datasets = load_dataset(args.dataset_name, split=args.dataset_config_name)
        else:
            raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name, split="validation")
    #TODO(Victor): enable loading pre-processed dataset from https://huggingface.co/datasets/bigscience/P3

    # Trim a number of evaluation examples
    if args.debug:
        raw_datasets = raw_datasets.select(range(100))

    column_names = raw_datasets.column_names
    """

    local_raw_data_dir = args.eval_data_dir
    dataset_names = [dataset_name for (dataset_name, _) in list(template_list.keys())]
    assert dataset_name in dataset_names

    data_path = os.path.join(local_raw_data_dir, dataset_name, dataset_config_name, 'validation')
    if os.path.exists(data_path):
        pass
    elif os.path.exists(os.path.join(local_raw_data_dir, dataset_name, dataset_config_name, 'valid')):
        os.path.join(local_raw_data_dir, dataset_name, dataset_config_name, 'valid')
    elif os.path.exists(os.path.join(local_raw_data_dir, dataset_name, dataset_config_name, 'test')):
        data_path=os.path.join(local_raw_data_dir, dataset_name, dataset_config_name, 'test')
    else:
        data_path = os.path.join(local_raw_data_dir, dataset_name, dataset_config_name, 'train')
    
    raw_datasets = load_from_disk(data_path)
    task_name='/'.join([dataset_name,dataset_config_name]) if dataset_config_name!='' else dataset_name
    raw_datasets = preprocess_dataset(raw_datasets,task_name)
    if len(raw_datasets)>args.max_valid_num:
        np.random.seed(42)
        idxs=range(len(raw_datasets))
        sample_idxs=np.random.choice(idxs,args.max_valid_num,False).tolist()
        raw_datasets=raw_datasets.select(sample_idxs)
    print('length of dataset',len(raw_datasets))
    # Trim a number of evaluation examples
    if args.debug:
        raw_datasets = raw_datasets.select(range(100))

    column_names = raw_datasets.column_names

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    padding = "max_length" if args.pad_to_max_length else False

    # Get the prompt to apply and the possible targets.
    # TODO(Victor): If pulling from pre-processed data, remove this logic.
    """
    prompts = DatasetTemplates(
        f"{args.dataset_name}"
        if args.dataset_config_name is None
        else f"{args.dataset_name}/{args.dataset_config_name}"
    )
    template = prompts[args.template_name]
    """

    if dataset_name == "anli":
        template_key = dataset_name
    elif dataset_name == "hellaswag":
        template_key = dataset_name
    else:
        template_key = f"{dataset_name}" if dataset_config_name is None else f"{dataset_name}/{dataset_config_name}"
    prompts = DatasetTemplates(template_key)
    template = prompts[template_name]

    def preprocess_function(examples):
        bs = len(examples[column_names[0]])

        input_texts = []
        target_texts = []
        answer_choices_texts = []
        for i in range(bs):

            ex = {k: examples[k][i] for k in column_names}

            outputs = template.apply(ex)

            if len(outputs) == 2:
                input, target = outputs
            else:
                assert (len(outputs) == 1 and len(outputs[0]) == 0)
                continue
                
            if len(outputs[1])==0: continue

            ex_answer_choices = template.get_answer_choices_list(ex)
            # import pdb
            # pdb.set_trace()
            # assert target in ex_answer_choices
            input_texts.append(input)
            target_texts.append(target)
            if ex_answer_choices is not None:
                answer_choices_texts.append(ex_answer_choices)
        variable_flag=-1 # -1: generation; 0: equal size answer_choice_list; 1: variable size answer_choice_list
        if len(answer_choices_texts)!=0:
            if len(set([len(x) for x in answer_choices_texts]))!=1:
                variable_flag=1
            else: variable_flag=0
            # assert len(set([len(x) for x in answer_choices_texts]))==1
        # import pdb 
        # pdb.set_trace()
        bs = len(input_texts)
        tokenized_inputs = tokenizer(
            input_texts,
            padding=padding,
            max_length=args.max_length,
            truncation=True,
            # add_special_tokens=False, 
        )

        if len(answer_choices_texts)!=0:
            tokenized_targets = [
                tokenizer(
                    ans_choi,
                    padding=True,
                    max_length=args.target_max_length,
                    truncation=True,
                )
                for ans_choi in answer_choices_texts
            ]
            # features = {
            #     k: [
            #         [elem for _ in range(len(tokenized_targets[idx]["input_ids"]))]
            #         for idx, elem in enumerate(v)
            #     ]
            #     for k, v in tokenized_inputs.items()
            # }
            # features["labels"] = [
            #     tokenized_targets[idx]["input_ids"]
            #     for idx in range(bs)
            # ]
            # features["labels_attention_mask"] = [
            #     tokenized_targets[idx]["attention_mask"]
            #     for idx in range(bs)
            # ]
            if variable_flag==0:
                features = {
                    k: [
                        [elem for _ in range(len(tokenized_targets[idx]["input_ids"]))]
                        for idx, elem in enumerate(v)
                    ]
                    for k, v in tokenized_inputs.items()
                }
                features["labels"] = [
                    tokenized_targets[idx]["input_ids"]
                    for idx in range(bs)
                ]
                features["labels_attention_mask"] = [
                    tokenized_targets[idx]["attention_mask"]
                    for idx in range(bs)
                ]
                try:
                    features["targets"] = [
                        answer_choices_texts[idx].index(t)
                        for idx, t in enumerate(target_texts)
                    ]
                except:
                    targets=[]
                    for idx,t in enumerate(target_texts):
                        for (i,choice) in enumerate(answer_choices_texts[idx]):
                            if t.startswith(choice): targets.append(i)
                    features['targets']=targets
            elif variable_flag==1:
                features = {
                    k: [
                        [elem] 
                        for idx, elem in enumerate(v) for _ in range(len(tokenized_targets[idx]["input_ids"]))
                    ]
                    for k, v in tokenized_inputs.items()
                }
                features["labels"] = [
                    [input_ids] for idx in range(bs) for input_ids in tokenized_targets[idx]["input_ids"]
                ]
                features["labels_attention_mask"] = [
                    [attention_mask] for idx in range(bs) for attention_mask in tokenized_targets[idx]["attention_mask"]
                ]
                features["targets"]=[target_texts[i] for (i,choi) in enumerate(answer_choices_texts) for j in range(len(choi))]
                features["answer_choices_ids"]=[i for (i,choi) in enumerate(answer_choices_texts) for j in range(len(choi))]
                features["answer_choices_texts"]=[choi for (i,choi) in enumerate(answer_choices_texts) for j in range(len(choi))]
            else:
                raise NotImplementedError
        else:
            # tokenized_targets = tokenizer(
            #     target_texts,
            #     padding=padding,
            #     max_length=args.max_length,
            #     truncation=True,
            #     # add_special_tokens=False, 
            # )
            features = {
                k: [
                    [elem]
                    for idx, elem in enumerate(v)
                ]
                for k, v in tokenized_inputs.items()
            }
            # import pdb 
            # pdb.set_trace()
            # features["targets"] = tokenized_targets["input_ids"]
            features["targets"]=target_texts
        # import pdb 
        # pdb.set_trace()
        return features

    with accelerator.main_process_first():
        eval_dataset = raw_datasets.map(
            preprocess_function, batched=True, remove_columns=column_names
        )

    eval_type='direct' if 'labels' not in eval_dataset[0] else 'constraint' if 'answer_choices_texts' not in eval_dataset[0] else 'variable_constraint'
    # import pdb 
    # pdb.set_trace()
    # Log a few random samples from the eval set:
    """
    for index in random.sample(range(len(eval_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {eval_dataset[index]}.")
    """
    # DataLoaders creation:
    if args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        data_collator = DataCollatorForMultipleChoice(
            tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None)
        )

    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)


    # Use the device given by the `accelerator` object.
    if not args.parallelize:
        model.to(accelerator.device)

    # Prepare everything with our `accelerator`.
    eval_dataloader = accelerator.prepare(eval_dataloader)


    # Metrics
    # metric = load_metric("accuracy")
    metric = load_metric("./eval_t0/accuracy.py")

    # Eval!
    total_batch_size = args.per_device_eval_batch_size * accelerator.num_processes

    logger.info("***** Running evaluation *****")
    logger.info(f"  Template Name = {template_name}")
    logger.info(f"  Num examples = {len(eval_dataset)}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_eval_batch_size}")
    logger.info(f"  Total eval batch size (w. parallel, distributed) = {total_batch_size}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(len(eval_dataloader)), disable=not accelerator.is_local_main_process)

    model.eval()
    preds=[];targets=[]
    for batch in eval_dataloader:
        # import pdb 
        # pdb.set_trace()
        # print(len(batch))
        if eval_type=='constraint':
            with torch.no_grad():
                predictions = model(batch)
            # print(len(predictions))
            preds.append(accelerator.gather(predictions).cpu().numpy())
            targets.append(accelerator.gather(batch['targets']).cpu().numpy())
            # metric.add_batch(
            #     predictions=accelerator.gather(predictions),
            #     references=accelerator.gather(batch["targets"]),
            # )
        elif eval_type=='variable_constraint':
            with torch.no_grad(): 
                predictions=model.get_constrained_logits(batch)
            preds.append(accelerator.gather(predictions).cpu().numpy())
        else:
            with torch.no_grad():
                predictions=model._model.generate(**batch,max_length=args.target_max_length)
            new_output=torch.zeros((predictions.shape[0],args.target_max_length),dtype=torch.long).to(accelerator.device)
            # print('len(predictions)',len(predictions))
            new_output[:,:predictions.shape[1]]=predictions
            predictions=accelerator.gather(new_output)
            preds+=[tokenizer.decode(output,skip_special_tokens=True) for output in predictions]
        # print(len(preds))
        # print(predictions)
        progress_bar.update(1)
    if eval_type=='constraint':
        preds=np.concatenate(preds)
        targets=np.concatenate(targets)
        preds=preds[:len(eval_dataloader.dataset)]
        targets=targets[:len(eval_dataloader.dataset)]
        eval_metric = metric._compute(preds,targets)
    elif eval_type=='variable_constraint':
        pred_logits=np.concatenate(preds)
        pred_logits=pred_logits[:len(eval_dataloader.dataset)]
        targets=[]
        rearanged_logits=[];answer_choices_texts=[]
        for cnt,(tgt,qas_id,choices) in enumerate(zip(eval_dataset['targets'],eval_dataset['answer_choices_ids'],eval_dataset['answer_choices_texts'])):
            if len(rearanged_logits)<=qas_id: 
                rearanged_logits.append([])
                targets.append(tgt)
                answer_choices_texts.append(choices)
            rearanged_logits[-1].append(pred_logits[cnt])
        preds=[];
        for qas_id,choices in enumerate(answer_choices_texts):
            max_id=np.argmax(rearanged_logits[qas_id])
            preds.append(choices[max_id])
        eval_metric=metric._compute(preds,targets)
    else:
        from accuracy import Metric_funcs
        preds=preds[:len(eval_dataloader.dataset)]
        targets=eval_dataset['targets']
        metric_funcs=['accuracy','f1']
        eval_metric={metric_name:[] for metric_name in metric_funcs}
        assert len(preds)==len(targets)
        for (pred,target) in zip(preds,targets):
            for metric_name in metric_funcs:
                eval_metric[metric_name].append(Metric_funcs[metric_name](target,pred))
        for metric_name in metric_funcs:
            if len(eval_metric[metric_name])!=0:
                eval_metric[metric_name]=np.mean(eval_metric[metric_name])

    accelerator.print(f"Result: {eval_metric}")

    """
    results = {
        "dataset_name": args.dataset_name,
        "dataset_config_name": args.dataset_config_name,
        "template_name": args.template_name,
        "evaluation": eval_metric
    }
    """
    results = {
        "dataset_name": dataset_name,
        "dataset_config_name": dataset_config_name,
        "template_name": template_name,
        "evaluation": eval_metric
    }

    if accelerator.is_main_process:
        if args.output_dir is not None:
            filename = dataset_name + "_" + dataset_config_name + "_results.json"
            with open(os.path.join(args.output_dir, filename), "a+") as f:
                json.dump(results, f, indent=4)
    return results


def main():
    args = parse_args()

    if args.debug:
        import pdb
        pdb.set_trace()

    # Initialize the accelerator. We will let the accelerator handle device placement for us.
    config={
        "distributed_type": "MULTI_GPU",
        "fp16": False,
        "machine_rank": 0,
        "main_process_ip": None,
        "main_process_port": None,
        "main_training_function": "main",
        "num_machines": 1,
        "num_processes": 8
    }
    accelerator = Accelerator(config)
    # accelerator = Accelerator()

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO)
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # Handle the output directory creation
    temp = os.path.split(os.path.normpath(args.eval_model_dir))[0]
    logger.info("\n\n\n")
    logger.info(os.path.split(temp)[1])
    logger.info(args.output_dir)
    temp='/'.join(args.eval_model_dir.split('/')[-4:])
    args.output_dir = os.path.join(args.output_dir, temp)
    # args.output_dir = os.path.join(args.output_dir, os.path.split(temp)[1])
    logger.info(args.output_dir)

    if accelerator.is_main_process:
        if args.reevaluate:
            shutil.rmtree(args.output_dir)
        os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    ### transform checkpoints 
    logger.info("\n\nTransforming ckpt!")
    inverse_transform_ckpt(args.eval_model_dir, args.model_name_or_path)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)

    if tokenizer.pad_token is None:
        for token in [tokenizer.eos_token, tokenizer.bos_token, tokenizer.sep_token]:
            if token is not None:
                tokenizer.pad_token = token
        if tokenizer.pad_token is None:
            raise ValueError("Please define a pad token id.")

    model = ModelBase.from_config(
        config=config,
        model_name_or_path=args.eval_model_dir,
        parallelize=args.parallelize
    )

    file_name=os.path.join(args.output_dir,'test_results.txt')
    already_task_names=[]
    if os.path.exists(file_name)==True:
        with open(file_name,'r') as f:
            lines=f.readlines()
            for line in lines:
                already_task_names.append(line.split('\t')[0])
        already_task_names=list(set(already_task_names))
    print('already_task_names',already_task_names)
    acc_results = {}
    for (dataset_name, dataset_config_name), prompt_list in template_list.items():
        prefix = dataset_name + "_" + dataset_config_name
        if prefix in already_task_names: continue
        f=open(os.path.join(args.output_dir,'test_results.txt'),'a+')
        ff=open(os.path.join(args.output_dir,'aranged_test_results.txt'),'a+')
        # acc_results[prefix] = dict()
        results=[]
        for template_name in prompt_list:
            cur_results = compute(args, model, tokenizer, dataset_name, dataset_config_name, template_name, accelerator)
            # acc_results[prefix]['accuracy'][template_name] = cur_results["evaluation"]['accuracy']
            results.append(cur_results['evaluation'])
        if accelerator.is_main_process:
            for (prompt_name,result) in zip(prompt_list,results):
                f.write('{}\t{}\t{}\n'.format(prefix,prompt_name,result))
                f.flush()
            results=rearange_results(results)
            ff.write('{}\t{}\n'.format(prefix,results))
            ff.flush()
        f.close()
        ff.close()
        # vals = list(acc_results[prefix].values())
        # keys = list(acc_results[prefix].keys())

        # acc_results[prefix]["mean_results"] = np.mean(vals)
        # acc_results[prefix]["median_results"] = np.median(vals)

        # logger.info("\n\n")
        # logger.info(f" Prefix:  {prefix}")
        # logger.info(f" Number of Prompts: {len(keys)}")
        # logger.info(f" Mean Results / Median Results: {np.mean(vals)}/{np.median(vals)}")
        # logger.info("\n\n")

    # fw_path = os.path.join(args.output_dir, "summary_results.json")
    # with open(fw_path, "w") as fw:
    #     json.dump(acc_results, fw, indent=4)

if __name__ == "__main__":
    main()