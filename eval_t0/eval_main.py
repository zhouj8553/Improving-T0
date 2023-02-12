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
from template_list import template_list

logger = logging.getLogger(__name__)


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
    args = parser.parse_args()

    return args


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

    split = "validation"
    data_path = os.path.join(local_raw_data_dir, dataset_name, dataset_config_name, split)
    raw_datasets = load_from_disk(data_path)

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

            ex_answer_choices = template.get_answer_choices_list(ex)
            assert target in ex_answer_choices
            input_texts.append(input)
            target_texts.append(target)
            answer_choices_texts.append(ex_answer_choices)

        bs = len(input_texts)

        tokenized_inputs = tokenizer(
            input_texts,
            padding=padding,
            max_length=args.max_length,
            truncation=True,
            # add_special_tokens=False, 
        )
        tokenized_targets = [
            tokenizer(
                ans_choi,
                padding=True,
                max_length=args.target_max_length,
                truncation=True,
            )
            for ans_choi in answer_choices_texts
        ]

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
        features["targets"] = [
            answer_choices_texts[idx].index(t)
            for idx, t in enumerate(target_texts)
        ]

        return features

    with accelerator.main_process_first():
        eval_dataset = raw_datasets.map(
            preprocess_function, batched=True, remove_columns=column_names
        )

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
        # with torch.no_grad():
        #     predictions = model(batch)
        with torch.no_grad():
            predictions = model(batch)
        # print(len(predictions))
        preds.append(accelerator.gather(predictions).cpu().numpy())
        targets.append(accelerator.gather(batch['targets']).cpu().numpy())
        # metric.add_batch(
        #     predictions=accelerator.gather(predictions),
        #     references=accelerator.gather(batch["targets"]),
        # )

        progress_bar.update(1)

    # eval_metric = metric.compute()
    preds=np.concatenate(preds)
    targets=np.concatenate(targets)
    preds=preds[:len(eval_dataloader.dataset)]
    targets=targets[:len(eval_dataloader.dataset)]
    eval_metric = metric._compute(preds,targets)
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


    acc_results = {}
    for (dataset_name, dataset_config_name), prompt_list in template_list.items():
        prefix = dataset_name + "_" + dataset_config_name
        acc_results[prefix] = dict()
        for template_name in prompt_list:
            cur_results = compute(args, model, tokenizer, dataset_name, dataset_config_name, template_name, accelerator)
            acc_results[prefix][template_name] = cur_results["evaluation"]["accuracy"]

        vals = list(acc_results[prefix].values())
        keys = list(acc_results[prefix].keys())

        acc_results[prefix]["mean_results"] = np.mean(vals)
        acc_results[prefix]["median_results"] = np.median(vals)

        logger.info("\n\n")
        logger.info(f" Prefix:  {prefix}")
        logger.info(f" Number of Prompts: {len(keys)}")
        logger.info(f" Mean Results / Median Results: {np.mean(vals)}/{np.median(vals)}")
        logger.info("\n\n")

    fw_path = os.path.join(args.output_dir, "summary_results.json")
    with open(fw_path, "w") as fw:
        json.dump(acc_results, fw, indent=4)

if __name__ == "__main__":
    main()