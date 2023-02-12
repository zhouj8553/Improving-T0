# coding=utf-8
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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

"""parses arguments and preps data loader"""

import os
import copy
import random
import numpy as np
import torch
import torch.utils.data
from tqdm import tqdm
import json
import random

import data_utils
from blocklm_utils import CollatorForGLM, CollatorForT5MLM
from tasks.p3.dataset import P3Dataset
from special_tasks_config import tasks_without_validation
from tasks.p3.p3 import datasets_without_validation, T0_TRAIN_TASK_LIST, T0_PLUS_TRAIN_TASK_LIST, \
    T0_PLUS_PLUS_TRAIN_TASK_LIST, DEBUG_TRAIN_TASK_LIST
from tasks.p3.p3 import REGISTERED_DATA_LIST
from utils import print_rank_0
from itertools import accumulate
from bisect import bisect_right
from tasks.superglue.dataset import SuperGlueDataset

from SwissArmyTransformer import mpu
from SwissArmyTransformer.tokenization import get_tokenizer
from data_utils import BertWordPieceTokenizer

# global_rng = random.Random(42)

def compute_input_and_target_lengths(inputs_length, noise_density, mean_noise_span_length):
    """This function is copy of `random_spans_helper <https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py#L2466>`__ .

    Training parameters to avoid padding with random_spans_noise_mask.
    When training a model with random_spans_noise_mask, we would like to set the other
    training hyperparmeters in a way that avoids padding.
    This function helps us compute these hyperparameters.
    We assume that each noise span in the input is replaced by extra_tokens_per_span_inputs sentinel tokens,
    and each non-noise span in the targets is replaced by extra_tokens_per_span_targets sentinel tokens.
    This function tells us the required number of tokens in the raw example (for split_tokens())
    as well as the length of the encoded targets. Note that this function assumes
    the inputs and targets will have EOS appended and includes that in the reported length.

    Args:
        inputs_length: an integer - desired length of the tokenized inputs sequence
        noise_density: a float
        mean_noise_span_length: a float
    Returns:
        tokens_length: length of original text in tokens
        targets_length: an integer - length in tokens of encoded targets sequence
    """

    def _tokens_length_to_inputs_length_targets_length(tokens_length):
        num_noise_tokens = int(round(tokens_length * noise_density))
        num_nonnoise_tokens = tokens_length - num_noise_tokens
        num_noise_spans = int(round(num_noise_tokens / mean_noise_span_length))
        # inputs contain all nonnoise tokens, sentinels for all noise spans
        # and one EOS token.
        _input_length = num_nonnoise_tokens + num_noise_spans + 1
        _output_length = num_noise_tokens + num_noise_spans + 1
        return _input_length, _output_length

    tokens_length = inputs_length

    while _tokens_length_to_inputs_length_targets_length(tokens_length + 1)[0] <= inputs_length:
        tokens_length += 1

    inputs_length, targets_length = _tokens_length_to_inputs_length_targets_length(tokens_length)

    # minor hack to get the targets length to be equal to inputs length
    # which is more likely to have been set to a nice round number.
    if noise_density == 0.5 and targets_length > inputs_length:
        tokens_length -= 1
        targets_length -= 1
    return tokens_length, targets_length


def make_tokenizer(args):
    outer_tokenizer = None
    if args.tokenizer_type == "glm_BertWordPieceTokenizer":
        outer_tokenizer = BertWordPieceTokenizer(tokenizer_model_type=args.tokenizer_model_type, add_block_symbols=True,
                                                 add_task_mask=args.task_mask,
                                                 add_decoder_mask=args.block_mask_prob > 0.0)
    tokenizer = get_tokenizer(args, outer_tokenizer=outer_tokenizer)
    args.eod_token = tokenizer.get_command('eos').Id
    return tokenizer

"""
class MultiTaskDataset(torch.utils.data.Dataset):
    def __init__(self, tasks, datasets, reweight=True, temperature=0.8, max_limit=200000):
        super(MultiTaskDataset, self).__init__()
        self.tasks = tasks
        self.datasets = datasets
        self.reweight = reweight
        self.temperature = temperature
        self.lens = [len(dataset) for dataset in datasets]
        self.weights = np.array([min(l, max_limit) ** temperature for l in self.lens])
        self.total_len = sum(self.lens)
        self.cumulative_lens = list(accumulate(self.lens))
        if self.reweight:
            print_rank_0(list(zip(self.tasks, self.lens, self.weights)))
        else:
            print_rank_0(list(zip(self.tasks, self.lens)))
        self.weights /= self.weights.sum()

    def __len__(self):
        return self.total_len * 1000

    @staticmethod
    def pet_wrapper(data):
        text = data['text']
        loss_mask = data['logit_mask']
        target = data['target']
        attention_mask = data['mask']
        position_id = data['position']
        label = data['label']
        if len(text.shape) == 2:
            text = text[label]
            loss_mask = loss_mask[label]
            target = target[label]
            attention_mask = attention_mask[label]
            position_id = position_id[label]
        else:
            target = target[label]
        if not target.shape:
            target = target.repeat(len(text))
        return {'text': text, 'target': target, 'loss_mask': loss_mask, 'position_id': position_id,
                'attention_mask': attention_mask}

    def __getitem__(self, idx):
        if self.reweight:
            rng = random.Random(idx)
            rng = np.random.RandomState(seed=[rng.randint(0, 2 ** 32 - 1) for _ in range(16)])
            dataset_idx = rng.choice(np.arange(len(self.datasets)), p=self.weights)
            dataset = self.datasets[dataset_idx]
            sample_idx = rng.choice(np.arange(len(dataset)))
            item = self.datasets[dataset_idx][sample_idx]
        else:
            dataset_idx = bisect_right(self.cumulative_lens, idx)
            if dataset_idx == 0:
                sample_idx = idx
            else:
                sample_idx = idx - self.cumulative_lens[dataset_idx - 1]
            item = self.datasets[dataset_idx][sample_idx]
        item = self.pet_wrapper(item)
        return item
"""

class MultiTaskDataset(torch.utils.data.Dataset):
    def __init__(self, task_list, datasets, reweight, temperature=0.8, max_limit=500000):
        super(MultiTaskDataset, self).__init__()
        self.task_list = task_list
        self.datasets = datasets
        self.reweight = reweight
        self.temperature = temperature
        self.lens = [len(dataset) for dataset in datasets]
        print_rank_0(self.lens)
        self.weights = np.array([min(l, max_limit) ** temperature for l in self.lens])
        self.total_len = sum(self.lens)
        print_rank_0(f"\n\n\nThe multiple dataset size: {self.total_len}.\n\n\n")
        self.cumulative_lens = list(accumulate(self.lens))
        if self.reweight:
            print_rank_0(list(zip(self.task_list, self.lens, self.weights)))
        else:
            print_rank_0(list(zip(self.task_list, self.lens)))
        self.weights /= self.weights.sum()

    def __len__(self):
        return self.total_len * 1000 if self.reweight else self.total_len

    def __getitem__(self, idx):
        if self.reweight:
            rng = random.Random(idx)
            rng = np.random.RandomState(seed=[rng.randint(0, 2 ** 32 - 1) for _ in range(16)])
            dataset_idx = rng.choice(np.arange(len(self.datasets)), p=self.weights)
            dataset = self.datasets[dataset_idx]
            sample_idx = rng.choice(np.arange(len(dataset)))
            item = self.datasets[dataset_idx][sample_idx]
        else:
            dataset_idx = bisect_right(self.cumulative_lens, idx)
            if dataset_idx == 0:
                sample_idx = idx
            else:
                sample_idx = idx - self.cumulative_lens[dataset_idx - 1]
            item = self.datasets[dataset_idx][sample_idx]
        return item



class DataConfig:

    def __init__(self, defaults=None):
        super(DataConfig, self).__init__()
        if defaults is None:
            defaults = {}
        self.defaults = defaults

    def apply(self, args, tokenizer):
        if torch.distributed.get_rank() == 0:
            print('configuring data')
        self.apply_defaults(args)
        return make_loaders(args, tokenizer)

    def set_defaults(self, **kwargs):
        for k, v in kwargs.items():
            self.defaults[k] = v

    def apply_defaults(self, args):
        for k, v in self.defaults.items():
            k = k.replace('-', '_')
            if not hasattr(args, k):
                setattr(args, k, v)


def make_data_loader(dataset, tokenizer, batch_size, num_iters, args, shuffle=False, collator=None):
    world_size = torch.distributed.get_world_size(group=mpu.get_data_parallel_group())
    rank = torch.distributed.get_rank(group=mpu.get_data_parallel_group())
    if args.loader_scatter is not None:
        loader_scatter = min(args.loader_scatter, mpu.get_data_parallel_world_size())
        rank = rank // loader_scatter
        world_size = world_size // loader_scatter
        batch_size = batch_size // loader_scatter
    distributed = world_size > 1
    if args.transformer_xl:
        batch_sampler = data_utils.samplers.DistributedSequentialSampler(len(dataset),
                                                                         num_iters,
                                                                         batch_size,
                                                                         rank,
                                                                         world_size)
    else:
        if shuffle:
            sampler = data_utils.samplers.RandomSampler(dataset, replacement=True,
                                                        num_samples=batch_size * args.train_iters * args.gradient_accumulation_steps)
        else:
            sampler = torch.utils.data.SequentialSampler(dataset)
        drop_last = distributed
        # the GPUs in the same model parallel group receive the same data
        if distributed:
            batch_sampler = data_utils.samplers.DistributedBatchSampler(sampler, batch_size, drop_last, rank,
                                                                        world_size,
                                                                        gradient_accumulation_steps=args.gradient_accumulation_steps)
        else:
            batch_sampler = torch.utils.data.BatchSampler(sampler,
                                                          batch_size,
                                                          drop_last)
    if collator == 'block':
        collate_fn = CollatorForGLM(args, tokenizer, args.seq_length, bert_prob=args.bert_prob,
                                    gap_sentence_prob=args.gap_sentence_prob,
                                    gap_sentence_ratio=args.gap_sentence_ratio,
                                    gpt_infill_prob=args.gpt_infill_prob,
                                    average_block_length=args.avg_block_length,
                                    gpt_min_ratio=args.gpt_min_ratio,
                                    block_mask_prob=args.block_mask_prob,
                                    context_mask_ratio=args.context_mask_ratio,
                                    short_seq_prob=args.short_seq_prob,
                                    single_span_prob=args.single_span_prob,
                                    shuffle_blocks=not args.no_shuffle_block,
                                    block_position_encoding=not args.no_block_position,
                                    sentinel_token=args.sentinel_token,
                                    encoder_decoder=args.encoder_decoder,
                                    task_mask=args.task_mask, random_position=args.random_position,
                                    masked_lm=args.masked_lm)
    elif collator == 't5':
        collate_fn = CollatorForT5MLM(tokenizer, noise_density=args.bert_mask_ratio,
                                      mean_noise_span_length=args.avg_block_length, prefix_lm=args.prefix_lm)
    elif collator is None:
        collate_fn = None
    else:
        raise NotImplementedError(collator)
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_sampler=batch_sampler,
                                              num_workers=args.num_workers,
                                              pin_memory=True,
                                              collate_fn=collate_fn)

    return data_loader


def make_loaders(args, tokenizer):
    """makes training/val/test"""
    world_size = torch.distributed.get_world_size(group=mpu.get_data_parallel_group())
    batch_size = args.batch_size * world_size
    eval_batch_size = batch_size
    if args.eval_batch_size is not None:
        eval_batch_size = args.eval_batch_size * world_size
    seq_length = args.seq_length
    eval_seq_length = args.eval_seq_length
    if args.t5_model:
        seq_length, _ = compute_input_and_target_lengths(seq_length, noise_density=args.bert_mask_ratio,
                                                      mean_noise_span_length=args.avg_block_length)
        print_rank_0(f"Using extended seq length {seq_length}")
        if eval_seq_length is not None:
            eval_seq_length, _ = compute_input_and_target_lengths(eval_seq_length, noise_density=args.bert_mask_ratio,
                                                               mean_noise_span_length=args.avg_block_length)
    split = get_split(args)
    data_set_args = {
        'path': args.train_data,
        'seq_length': seq_length,
        'mem_length': args.mem_length,
        'ds_type': args.data_set_type,
        'split': split,
        'dataset_temperature': args.dataset_temperature,
        'sample_one_document': args.sample_one_document,
        'filter_english': args.filter_english,
        'pre_tokenize': not args.no_pre_tokenize,
        'tokenizer': tokenizer,
        'save_splits': args.save_splits,
        'load_splits': args.load_splits,
        'save_test_data': args.save_test_data,
        'no_lazy_loader': args.no_lazy_loader,
        'loader_scatter': args.loader_scatter,
        'data_parallel_rank': mpu.get_data_parallel_rank(),
        "non_sentence_start": args.non_sentence_start,
        "loader_fraction": args.loader_fraction
    }
    if args.t5_model:
        data_set_args["add_cls"] = False
        data_set_args["add_eos"] = False
        data_set_args["sentence_end"] = False

    eval_set_args = copy.copy(data_set_args)
    eval_set_args['split'] = [1.]
    # if optional eval args were set then replace their
    # equivalent values in the arg dict
    if eval_seq_length:
        eval_set_args['seq_length'] = eval_seq_length

    # make datasets splits and tokenizer
    train, valid, test = None, None, None

    if args.train_data is not None:
        train = data_utils.make_dataset(**data_set_args)
        if data_utils.should_split(split):
            train, valid, test = train
        eval_set_args['tokenizer'] = tokenizer

    # make training and val dataset if necessary
    if valid is None and args.valid_data is not None:
        eval_set_args['path'] = args.valid_data
        valid = data_utils.make_dataset(**eval_set_args)
        eval_set_args['tokenizer'] = tokenizer
    if test is None and args.test_data is not None:
        eval_set_args['path'] = args.test_data
        test = data_utils.make_dataset(**eval_set_args)

    # wrap datasets with data loader
    if args.block_lm or args.encoder_decoder:
        collator = 'block'
    elif args.t5_model:
        collator = 't5'
    else:
        collator = None

    if train is not None and args.batch_size > 0:
        train = make_data_loader(train, tokenizer, batch_size, args.train_iters, args, shuffle=args.shuffle,
                                 collator=collator)
        args.do_train = True
    else:
        args.do_train = False
    eval_batch_size = eval_batch_size if eval_batch_size != 0 else batch_size
    if valid is not None:
        valid = make_data_loader(valid, tokenizer, eval_batch_size, args.train_iters, args, shuffle=args.shuffle,
                                 collator=collator)
        args.do_valid = True
    else:
        args.do_valid = False
    if test is not None:
        test = make_data_loader(test, tokenizer, eval_batch_size, len(test) // eval_batch_size + 1, args,
                                shuffle=args.shuffle, collator=collator)
        args.do_test = True
    else:
        args.do_test = False

    return train, valid, test


def check_task_without_validation(task_name):
    for prefix in datasets_without_validation:
        prefix = prefix.replace("/", "_")
        if task_name.startswith(prefix):
            return True
    return False

def check_task_without_validation_test(task_name):
    if task_name in tasks_without_validation:
        return True
    return False


def build_multi_task_dataset(args, tokenizer):
    if args.t0_prepared_task_names is not None:
        t0_task_names=[]
        for dataset_name in args.t0_prepared_task_names:
            if dataset_name in REGISTERED_DATA_LIST:
                t0_task_names+=REGISTERED_DATA_LIST[dataset_name]
            else:
                t0_task_names.append(dataset_name)
    else:
        if args.multi_task_set == "t0":
            # t0_task_names = ls_0525_1 # combine_ls_0513
            t0_task_names=T0_TRAIN_TASK_LIST
        elif args.multi_task_set == "t0+":
            t0_task_names = T0_PLUS_TRAIN_TASK_LIST
        elif args.multi_task_set == "t0++":
            t0_task_names = T0_PLUS_PLUS_TRAIN_TASK_LIST
        elif args.multi_task_set == "qa":
            t0_task_names = T0_TRAIN_TASK_LIST
        elif args.multi_task_set == "debug":
            t0_task_names = DEBUG_TRAIN_TASK_LIST
        else:
            raise ValueError(f"Unknown multi-task-set {args.multi_task_set}.")
    t0_task_names = [item for item in t0_task_names if not item.endswith("score_eval")]

    train_datasets, valid_datasets = [], []
    for task_name in tqdm(t0_task_names):
        """
        train_dataset = P3Dataset(args, task_name, "train", tokenizer, is_training=True)
        with open(os.path.join("/dataset/fd5061f6/yanan/data/packed_preprocessed_t0", task_name, "train.json"),
                  "w") as train_file:
            for example in train_dataset.example_list:
                data = {"input": example.text_a, "target": example.text_b, "guid": example.guid}
                data_str = json.dumps(data)
                train_file.write(data_str + "\n")

        if check_task_without_validation(task_name):
            valid_dataset = P3Dataset(args, task_name, "test", tokenizer, is_training=True)
        else:
            valid_dataset = P3Dataset(args, task_name, "validation", tokenizer, is_training=True)
        with open(os.path.join("/dataset/fd5061f6/yanan/data/packed_preprocessed_t0", task_name, "validation.json"),
                  "w") as valid_file:
            for example in valid_dataset.example_list:
                data = {"input": example.text_a, "target": example.text_b, "guid": example.guid}
                data_str = json.dumps(data)
                valid_file.write(data_str + "\n")
        train_datasets.append(train_dataset)
        valid_datasets.append(valid_dataset)
        """
        # all the task_name here are prompted
        global_rng = random.Random(42)
        train_datasets.append(P3Dataset(args, task_name, "train", tokenizer, is_training=True, rng=global_rng)) # 0317
        # if check_task_without_validation(task_name):
        #     valid_datasets.append(P3Dataset(args, task_name, "test", tokenizer, is_training=True, rng=global_rng)) # 0317
        # else:
        #     valid_datasets.append(P3Dataset(args, task_name, "validation", tokenizer, is_training=True, rng=global_rng)) # 0317
        # if task_name in ablation_ls: # 0506
        #     print("Ablation study")
        #     valid_datasets.append(P3Dataset(args, task_name, "validation", tokenizer, is_training=True, rng=global_rng)) # 0317
        if task_name.startswith('app_reviews'):
            pass
        elif check_task_without_validation_test(task_name): # 0423 add zj tasks
            print("ZJ's tasks")
        elif check_task_without_validation(task_name):
            valid_datasets.append(P3Dataset(args, task_name, "test", tokenizer, is_training=True, rng=global_rng)) # 0317
        else:
            valid_datasets.append(P3Dataset(args, task_name, "validation", tokenizer, is_training=True, rng=global_rng)) # 0317
        

    train = MultiTaskDataset(t0_task_names, train_datasets, reweight=args.reweight)
    valid = MultiTaskDataset(t0_task_names, valid_datasets, reweight=args.reweight)
    world_size = torch.distributed.get_world_size(group=mpu.get_data_parallel_group())

    multi_batch_size = args.batch_size * world_size
    if args.multi_batch_size is not None:
        multi_batch_size = args.multi_batch_size * world_size
    multi_eval_batch_size = args.batch_size * world_size
    if args.eval_batch_size is not None:
        multi_eval_batch_size = args.eval_batch_size * world_size

    print("\n\n")
    print("****************************")
    print(f"train_data_size_per_rank: {len(train)}.")
    print(f"total_data_size: {len(train) * 8}.")
    print(f"train_steps: {args.train_iters}.")
    print(f"gradient_accumulation_steps: {args.gradient_accumulation_steps}.")
    print(f"Batch Size per gpu: {args.batch_size}.")
    print(f"multi_batch_size: {multi_batch_size}")
    print(f"multi_eval_batch_size: {multi_eval_batch_size}")
    print("\n")
    print(int(len(train) * 8 / (multi_batch_size * args.gradient_accumulation_steps)))
    print(f"less than {args.train_iters} is true???")
    print("\n")
    # assert int(len(train) * 8 / (multi_batch_size * args.gradient_accumulation_steps)) < args.train_iters

    shuffle = False if args.reweight else True
    train = make_data_loader(train, tokenizer, multi_batch_size, args.train_iters, args, shuffle=shuffle,  collator=None)
    valid = make_data_loader(valid, tokenizer, multi_eval_batch_size, args.train_iters, args,
                             shuffle=shuffle, collator=None)
    return train, valid


"""
def build_multi_task_dataset(args, tokenizer):
    task_dirs = {"mnli": "MNLI", "cola": "CoLA", "mrpc": "MRPC", "qnli": "QNLI", "qqp": "QQP", "sst2": "SST-2",
                 "agnews": "Agnews", "yelp-polarity": "yelp_review_polarity_csv", "yelp-full": "yelp_review_full_csv",
                 "yahoo": "Yahoo", "squad": "SQuAD", "race": "RACE"}
    train, valid = None, None
    if mpu.get_model_parallel_rank() == 0:
        multi_seq_length = args.seq_length
        if args.multi_seq_length is not None:
            multi_seq_length = args.multi_seq_length
        train_datasets, valid_datasets = [], []
        for task in args.multi_task_data:
            task = task.lower()
            data_dir = os.path.join(args.data_dir, task_dirs[task])
            train_datasets.append(
                SuperGlueDataset(args, task, data_dir, multi_seq_length, "train", tokenizer, pattern_ensemble=True))
            valid_datasets.append(
                SuperGlueDataset(args, task, data_dir, multi_seq_length, "dev", tokenizer, pattern_ensemble=True))
        train = MultiTaskDataset(args.multi_task_data, train_datasets)
        valid = MultiTaskDataset(args.multi_task_data, valid_datasets)
        world_size = torch.distributed.get_world_size(group=mpu.get_data_parallel_group())
        multi_batch_size = args.batch_size * world_size
        if args.multi_batch_size is not None:
            multi_batch_size = args.multi_batch_size * world_size
        train = make_data_loader(train, tokenizer, multi_batch_size, args.train_iters, args, shuffle=True)
        valid = make_data_loader(valid, tokenizer, multi_batch_size, args.train_iters, args, shuffle=True)
    return train, valid
"""

def get_split(args):
    """
    Get dataset splits from comma separated string list
    """
    splits = []
    if args.split.find(',') != -1:
        splits = [float(s) for s in args.split.split(',')]
    elif args.split.find('/') != -1:
        splits = [float(s) for s in args.split.split('/')]
    else:
        splits = [float(args.split)]
    split_total = sum(splits)
    if split_total < 1.:
        splits.append(1 - split_total)
    while len(splits) < 3:
        splits.append(0.)
    splits = splits[:3]
    if args.valid_data is not None:
        splits[1] = 0.
    if args.test_data is not None:
        splits[2] = 0.
    final_sum = sum(splits)
    return [s / final_sum for s in splits]


def configure_data():
    """add cmdline flags for configuring datasets"""
    # These are options that are used by data_utils, but are either
    # deprecated or not meant to be exposed to the command line user.
    # These options are intneded to be set in code by specific scripts.
    defaults = {
        'world_size': 1,
        'rank': -1,
        'persist_state': 0,
        'lazy': False,
        'transpose': False,
        'data_set_type': 'supervised',
        'seq_length': 256,
        'eval_seq_length': 256,
        'samples_per_shard': 100
    }

    return DataConfig(defaults=defaults)
