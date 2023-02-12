import json
import os
import time

import numpy as np
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from tqdm import tqdm

from data_utils import exists_lazy, LazyLoader, LazyWriter
from tasks.data_utils import InputExample
from tasks.p3.p3 import large_t0_task_dict
from tasks.p3.pvp import P3PVP
from tasks.superglue.dataset import TEST_SET
from utils import print_rank_0
from SwissArmyTransformer import mpu


class DataProcessor:
    def __init__(self, args, task_name, tokenizer, lazy_seq2seq_loader=False, **kwargs):
        self.args = args
        self.data_dir = args.multi_cache_dir
        self.max_src_len = self.args.multi_src_seq_length
        self.max_tgt_len = self.args.multi_tgt_seq_length
        self.task_name = task_name
        self.tokenizer = tokenizer
        self.lazy_seq2seq_loader = lazy_seq2seq_loader

    def _yield_examples(self, split, dataset):
        raise NotImplementedError

    def create_examples(self, split, selected=True):
        print_rank_0(f"Creating {split} dataset from {self.data_dir} for task {self.task_name}.")
        if not self.lazy_seq2seq_loader:
            assert self.args.loader_scatter == 8, "--loader_scatter should be fixed to be 8."
            current_idx = mpu.get_data_parallel_rank() % 8 # [0,1,2,3,4,5,6,7]
            filepath = os.path.join(self.data_dir, self.task_name, split + ".json")
            print_rank_0(self.task_name)
            if selected:
                start = int(current_idx/8 * 100)
                end = int((current_idx + 1)/8 * 100)
                print_rank_0(split + f"[{start}%:{end}%]")
                split_str = split + f"[{start}%:{end}%]"
            elif self.task_name in list(large_t0_task_dict.keys()) and split == "train":
                total_num = large_t0_task_dict[self.task_name]
                start_number = int(current_idx/8 * total_num)
                end_number = int((current_idx + 1)/8 * total_num)
                print_rank_0(f"{split}[{start_number}:{end_number}]")
                split_str = f"{split}[{start_number}:{end_number}]"
            else:
                start = int(current_idx/8 * 100)
                end = int((current_idx + 1)/8 * 100)
                print_rank_0(split + f"[{start}%:{end}%]")
                split_str = split + f"[{start}%:{end}%]"
            print(f"Begin loading file:{filepath}")
            random_sample = False
            # dataset = load_dataset("json", data_files={split: filepath}, split=split_str,cache_dir="/share/zongyu/cache/huggingface/datasets")
            dataset = load_dataset("json", data_files={split: filepath}, split=split_str)
            if random_sample and split == "train":
                seed_ = 58 # 42,58,30,24,10
                sample_ratio = 0.9
                dataset.shuffle(seed=seed_)
                data_size = len(dataset)
                sample_num = int(sample_ratio*data_size)
                dataset = dataset.select(list(range(sample_num)))

            example_list = []
            for idx, example in enumerate(self._yield_examples(split, dataset)):
                if (idx + 1) % 20000 == 0:
                    print_rank_0(f"Complete {idx + 1} examples")
                example_list.append(example)
        else:
            """
            if (not exists_lazy(self.data_dir, data_type=split) and torch.distributed.get_rank() == 0):
                example_writer = LazyWriter(self.data_dir, data_type=split, is_array=False)
                for idx, example in enumerate(self._yield_examples(split)):
                    if (idx + 1) % 20000 == 0:
                        print_rank_0(f"Complete {idx + 1} examples")
                    example_writer.write(example)
            else:
                while not os.path.exists(LazyWriter.get_len_path(self.data_dir, data_type=split)):
                    time.sleep(1)
            example_list = LazyLoader(self.data_dir, data_type=split, map_fn=InputExample.from_json_string,
                                      mem_map=True, is_array=False)
            """
            raise NotImplementedError("lazy_seq2seq_loader not implemented.")

        print_rank_0(f"Creating {len(example_list)} examples for {split} of task {self.task_name}.")
        return example_list

class P3Processor(DataProcessor):
    def _yield_examples(self, split, dataset):
        source_texts, target_texts = [], []
        assert "inputs_pretokenized" in dataset.features
        assert "targets_pretokenized" in dataset.features
        for data_example in dataset:
            source_text = data_example["inputs_pretokenized"]
            source_texts.append(source_text)
            target_text = data_example["targets_pretokenized"]
            target_texts.append(target_text)
        assert len(source_texts) == len(target_texts)

        eos_token = self.tokenizer.get_command('eos').token

        def exceed_maximum_length(prev_inputs, inputs, max_seq_len, is_source, mask_token_index):
            assert isinstance(prev_inputs, str) and isinstance(inputs, str)
            prev_tok = self.tokenizer.EncodeAsIds(prev_inputs).tokenization
            assert len(prev_tok) <= max_seq_len
            tok = self.tokenizer.EncodeAsIds(inputs).tokenization

            if len(tok) >= max_seq_len - 2:
                tok = tok[:(max_seq_len - 2)]
                inputs = self.tokenizer.DecodeIds(tok)

            if self.args.t0_format == "lm_format":
                if len(prev_tok) + len(tok) < max_seq_len - 2:
                    ret_inputs = prev_inputs + (inputs + eos_token)
                    return False, ret_inputs
                else:
                    ret_inputs = prev_inputs
                    return True, ret_inputs

            elif self.args.t0_format == "denoise_format":
                if len(prev_tok) + len(tok) < max_seq_len - 2:
                    mask_token = self.tokenizer.get_command(f"MASK{mask_token_index}").token
                    if is_source:
                        ret_inputs = prev_inputs + (inputs + mask_token + eos_token)
                        return False, ret_inputs
                    else:
                        ret_inputs = prev_inputs + (mask_token + inputs + eos_token)
                        return False, ret_inputs
                else:
                    ret_inputs = prev_inputs
                    return True, ret_inputs
            else:
                raise ValueError("Unknown format.")

        if not self.args.packing:
            if self.args.t0_format == "lm_format":
                for idx, (source_text, target_text) in enumerate(zip(source_texts, target_texts)):
                    guid = "%s-%s" % (split, idx)
                    meta = {"ref": self.tokenizer.DecodeIds(self.tokenizer.EncodeAsIds(target_text).tokenization)}
                    example = InputExample(guid=guid, text_a=source_text, text_b=target_text, meta=meta)
                    if idx < 3:
                        print_rank_0((source_text.encode('utf-8'), target_text.encode('utf-8'), meta["ref"].encode('utf-8')))
                    yield example
            elif self.args.t0_format == "denoise_format":
                raise NotImplementedError("Not implemented denoise_format for non-packing.")
            else:
                raise NotImplementedError("Not implemented format.")
        else:
            print_rank_0("Packing data.")
            assert self.args.t5_model, "Only implement packing for T5Model."
            packed_source_texts, packed_target_texts = [], []
            cur_src_texts = ""
            cur_tgt_texts = ""
            mask_index = 0
            for source_text, target_text in tqdm(zip(source_texts, target_texts)):
                src_flag, temp_src = exceed_maximum_length(cur_src_texts, source_text, self.max_src_len,
                                                           is_source=True, mask_token_index=mask_index)
                tgt_flag, temp_tgt = exceed_maximum_length(cur_tgt_texts, target_text, self.max_tgt_len,
                                                           is_source=False, mask_token_index=mask_index)
                if (not src_flag) and (not tgt_flag):
                    cur_src_texts = temp_src
                    cur_tgt_texts = temp_tgt
                    mask_index += 1
                else:
                    packed_source_texts.append(cur_src_texts)
                    packed_target_texts.append(cur_tgt_texts)
                    mask_index = 0
                    _, cur_src = exceed_maximum_length("", source_text, self.max_src_len,is_source=True,
                                                       mask_token_index = mask_index)
                    _, cur_tgt = exceed_maximum_length("", target_text, self.max_tgt_len,is_source=False,
                                                       mask_token_index = mask_index)
                    cur_src_texts = cur_src
                    cur_tgt_texts = cur_tgt

            for idx, (source_text, target_text) in enumerate(zip(packed_source_texts, packed_target_texts)):
                guid = "%s-%s" % (split, idx)
                meta = {"ref": self.tokenizer.DecodeIds(self.tokenizer.EncodeAsIds(target_text).tokenization)}
                example = InputExample(guid=guid, text_a=source_text, text_b=target_text, meta=meta)
                if idx < 3:
                    print_rank_0((source_text.encode('utf-8'), target_text.encode('utf-8'), meta["ref"].encode('utf-8')))
                yield example



class P3Dataset(Dataset):
    def __init__(self, args, task_name, split, tokenizer, is_training=True):
        self.args = args
        self.task = task_name
        self.t5_model = args.t5_model
        self.max_src_length, self.max_tgt_length = args.multi_src_seq_length, args.multi_tgt_seq_length
        self.split = split
        self.tokenizer = tokenizer
        self.dataset_name = split
        self.is_training = is_training

        self.processor = P3Processor(self.args, self.task, tokenizer, lazy_seq2seq_loader=False)
        example_list = self.processor.create_examples(split)
        self.example_list = example_list
        self.examples = {example.guid: example for example in example_list}
        print_rank_0(f"Return {len(self.examples)} {split} examples for task {task_name}.")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.example_list[idx]
        pad_id = self.tokenizer.get_command('pad').Id
        sop_id = self.tokenizer.get_command('sop').Id

        if self.t5_model:
            eos_id = self.tokenizer.get_command('eos').Id
            source_text, target_text = example.text_a, example.text_b

            if not self.args.packing:
                source_tokens = self.tokenizer.EncodeAsIds(source_text).tokenization
                if len(source_tokens) > self.max_src_length - 1:
                    source_tokens = source_tokens[: (self.max_src_length - 1)]
                source_tokens = source_tokens + [eos_id]
            else:
                source_tokens = self.tokenizer.EncodeAsIds(source_text).tokenization

            attention_mask = [1] * len(source_tokens)
            if len(source_tokens) < self.max_src_length:
                pad_length = self.max_src_length - len(source_tokens)
                source_tokens = source_tokens + [pad_id] * pad_length
                attention_mask = attention_mask + [0] * pad_length

            # if self.split == "train" or self.split == "validation":
            if self.is_training:
                if not self.args.packing:
                    target_tokens = self.tokenizer.EncodeAsIds(target_text).tokenization
                    if len(target_tokens) > self.max_tgt_length - 1:
                        target_tokens = target_tokens[: (self.max_tgt_length - 1)]
                    target_tokens = target_tokens + [eos_id]
                else:
                    target_tokens = self.tokenizer.EncodeAsIds(target_text).tokenization

                loss_mask = [1] * len(target_tokens)
                if len(target_tokens) < self.max_tgt_length:
                    pad_length = self.max_tgt_length - len(target_tokens)
                    target_tokens = target_tokens + [pad_id] * pad_length
                    loss_mask = loss_mask + [0] * pad_length

                sample = {'text': np.array(source_tokens, dtype=np.int64),
                          'target': np.array(target_tokens, dtype=np.int64),
                          'attention_mask': np.array([[attention_mask]], dtype=np.int64),
                          'loss_mask': np.array(loss_mask, dtype=np.int64),
                          "uid": example.guid}
            else:### TODO: test sets of training tasks are not used.
                sample = {
                    'text': np.array(source_tokens, dtype=np.int64),
                    'attention_mask': np.array([[attention_mask]], dtype=np.int64),
                    "uid": example.guid
                }
        else:
            eop_id = self.tokenizer.get_command('eop').Id
            pvp = P3PVP(self.tokenizer,
                        max_src_length=self.max_src_length,
                        max_tgt_length=self.max_tgt_length,
                        task_mask=self.args.task_mask)
            mask_id = pvp.mask_id
            source_tokens, target_text = pvp.encode(example)

            if len(source_tokens) < self.max_src_length:
                source_tokens = source_tokens + [pad_id] * (self.max_src_length - len(source_tokens))
            sep = len(source_tokens)
            position_ids = list(range(len(source_tokens)))
            block_position_ids = [0] * len(source_tokens)
            mask_pos = source_tokens.index(mask_id)

            # if self.split == 'train' or self.split == "validation":
            if self.is_training:
                target_tokens = self.tokenizer.EncodeAsIds(" " + target_text).tokenization
                target_tokens = target_tokens + [eop_id]

                if len(target_tokens) > self.max_tgt_length:
                    target_tokens = target_tokens[:self.max_tgt_length]
                loss_mask = [1] * len(target_tokens)

                if len(target_tokens) < self.max_tgt_length:
                    loss_mask += [0] * (self.max_tgt_length - len(target_tokens))
                    target_tokens += [pad_id] * (self.max_tgt_length - len(target_tokens))

                tokens = source_tokens + [sop_id] + target_tokens[:-1]
                loss_mask = [0] * len(source_tokens) + loss_mask
                target_ids = [0] * len(source_tokens) + target_tokens
                position_ids += [mask_pos] * len(target_tokens)
                if self.args.no_block_position:
                    block_position_ids += [1] * len(target_tokens)
                else:
                    block_position_ids += list(range(1, len(target_tokens) + 1))
                position_ids = [position_ids, block_position_ids]
                sample = {'text': np.array(tokens, dtype=np.int64),
                          'target': np.array(target_ids, dtype=np.int64),
                          'attention_mask': np.array(sep, dtype=np.int64),
                          'loss_mask': np.array(loss_mask, dtype=np.int64),
                          "position_id": np.array(position_ids, dtype=np.int64),
                          "uid": example.guid}
            else:
                tokens = source_tokens + [sop_id]
                position_ids = position_ids + [mask_pos]
                block_position_ids = block_position_ids + [1]
                position_ids = [position_ids, block_position_ids]
                sample = {'text': np.array(tokens, dtype=np.int64),
                          'attention_mask': np.array(sep, dtype=np.int64),
                          "position_id": np.array(position_ids, dtype=np.int64),
                          "uid": example.guid}
        return sample
