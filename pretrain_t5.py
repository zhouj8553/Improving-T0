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

"""Pretrain T5 Model"""

# Flag to use Pytorch ddp which uses overlapping communication and computation.
from datetime import datetime
import os
import random
import math
import argparse

import deepspeed
import torch.distributed
from filelock import FileLock
import torch
from transformers import Adafactor
from transformers.optimization import AdafactorSchedule

from arguments import get_args
from configure_data import configure_data, build_multi_task_dataset, make_tokenizer
import pathlib

from utils import Timers
from utils import save_checkpoint, load_checkpoint
from utils import print_and_save_args, print_rank_0, get_sample_writer, get_log_dir
from SwissArmyTransformer.training.deepspeed_training import initialize_distributed, \
    set_random_seed, setup_model_and_optimizer, get_model, get_optimizer_param_groups
from SwissArmyTransformer import mpu
from SwissArmyTransformer.model import T5Model
from learning_rates import get_learning_rate_scheduler
from train_utils import evaluate_and_print_results, train, get_train_val_test_data
from model.prompt import get_prefix_model

from move_t5 import convert_hf_to_ds


def decoder_shift_right(input_ids, args):
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
    shifted_input_ids[..., 0] = args.decoder_start_token_id
    return shifted_input_ids


def get_batch(data, args):
    keys = ['text', 'loss_mask', 'target', 'attention_mask']
    datatype = torch.int64

    # Broadcast data.
    data_b = mpu.broadcast_data(keys, data, datatype)
    # Unpack.
    tokens = data_b['text'].long()
    labels = data_b['target'].long()
    decoder_tokens = decoder_shift_right(labels, args)
    attention_mask = data_b['attention_mask'].long()
    loss_mask = data_b['loss_mask'].float()

    # Convert
    if args.fp16:
        attention_mask = attention_mask.half()
    elif args.bf16:
        attention_mask = attention_mask.bfloat16()
    return tokens, decoder_tokens, labels, loss_mask, attention_mask


def forward_step(data_iterator, model, args, timers):
    """Forward step."""

    # Get the batch.
    timers('batch generator').start()
    timers('data loader').start()
    rand = random.Random(args.iteration * mpu.get_data_parallel_world_size() + mpu.get_data_parallel_rank())
    if data_iterator[1] and rand.random() < args.multi_task_ratio:
        data = next(data_iterator[1]) if data_iterator[1] else None
        data["mode"] = "multi-task"
    else:
        data = next(data_iterator[0]) if data_iterator[0] else None
    # print_rank_0("data iterator")
    timers('data loader').stop()
    tokens, decoder_tokens, labels, loss_mask, attention_mask = get_batch(data, args)
    timers('batch generator').stop()

    if data is not None and "mode" in data:
        mode = data['mode']
    else:
        mode = 'bert'

    _, logits, *_ = model(enc_input_ids=tokens, dec_input_ids=decoder_tokens, enc_attention_mask=attention_mask)
    losses = mpu.vocab_parallel_cross_entropy(logits.contiguous().float(), labels)
    loss_mask = loss_mask.view(-1)
    loss = torch.sum(losses.view(-1) * loss_mask)
    if loss_mask.sum().item() > 0:
        loss = loss / loss_mask.sum()
    metrics = {name: torch.cuda.FloatTensor([1]) if name == mode else torch.cuda.FloatTensor([0]) for name in
               ['bert', 'sentence', 'gpt', 'multi-task']}
    return loss, metrics


def main(args):
    """Main training program."""

    # Disable CuDNN.
    torch.backends.cudnn.enabled = False
    # Timer.
    timers = Timers()

    if args.load and not args.new_save_directory:
        args.experiment_name = os.path.basename(os.path.normpath(args.load))
    else:
        args.experiment_name = args.experiment_name + datetime.now().strftime("%m-%d-%H-%M")
    if args.save:
        args.save = os.path.join(args.save, args.experiment_name)
    # Pytorch distributed.
    print('start initialize')
    initialize_distributed(args)
    print('end initialize')
    # Random seeds for reproducability.
    set_random_seed(args.seed)

    # Data stuff.
    tokenizer = make_tokenizer(args)
    args.decoder_start_token_id = tokenizer.get_command('sop').Id

    # train_data, val_data, test_data, = get_train_val_test_data(args, tokenizer)
    train_data, val_data, test_data = None, None, None
    multi_train_data, multi_val_data = None, None 
    if args.multi_task_ratio > 0.0:
        multi_train_data, multi_val_data = build_multi_task_dataset(args, tokenizer)
    torch.distributed.barrier()

    # Model, optimizer, and learning rate.
    model_cls = get_prefix_model(T5Model) if args.prefix_prompt > 0 else T5Model

    if args.use_adafactor:
        # TODO: add adafactor
        print("Using Adafactor.")
        model = get_model(args, model_cls)
        model.disable_untrainable_params()
        param_groups = get_optimizer_param_groups(model)
        print_rank_0(f"Current Learning Rate: {args.lr}.")
        optimizer = Adafactor(param_groups, scale_parameter=False, relative_step=False, warmup_init=False, lr=args.lr)
        lr_scheduler = AdafactorSchedule(optimizer, initial_lr=args.lr) if optimizer is not None else None
        if args.train_data is not None:
            if args.deepspeed:
                print_rank_0("DeepSpeed is enabled.")
                model, optimizer, _, _ = deepspeed.initialize(
                    model=model,
                    optimizer=optimizer,
                    args=args,
                    mpu=mpu,
                    dist_init_required=False)
            else:
                raise ValueError('Currently, we only support training with deepspeed.')
        else:
            optimizer = None
    else:
        print("Using default Adam.")
        model, optimizer = setup_model_and_optimizer(args, model_cls=model_cls)
        lr_scheduler = get_learning_rate_scheduler(optimizer, args) if optimizer is not None else None

    if args.load is not None:
        with FileLock(os.path.join(pathlib.Path.home(), "checkpoint_lock"), timeout=-1):
            args.iteration = load_checkpoint(model, optimizer, lr_scheduler, args, no_deepspeed=args.no_deepspeed_load)
        if args.no_load_lr_scheduler:
            lr_scheduler.num_iters = args.iteration
    else:
        args.iteration = 0
    torch.distributed.barrier()
    if args.switch_linear:
        lr_scheduler.switch_linear(args)

    summary_writer = None
    if torch.distributed.get_rank() == 0:
        print('Pretrain GPT2 model')
        args.log_dir = None
        if args.train_iters > 0:
            args.log_dir = get_log_dir(base=args.summary_dir, name=args.experiment_name)
            summary_writer = get_sample_writer(log_dir=args.log_dir, iteration=args.iteration)
        print_and_save_args(args, verbose=True, log_dir=args.log_dir)

    # Resume data loader if necessary.
    if args.resume_dataloader:
        print_rank_0("Resume dataloader")
        if train_data is not None:
            train_data.batch_sampler.start_iter = args.iteration % len(train_data)
        if val_data is not None:
            start_iter_val = (args.iteration // args.eval_interval) * args.eval_iters
            val_data.batch_sampler.start_iter = start_iter_val % len(val_data)
        if multi_train_data is not None:
            multi_train_data.batch_sampler.start_iter = int(args.iteration * args.multi_task_ratio) % len(
                multi_train_data)
        if multi_val_data is not None:
            start_iter_val = (args.iteration // args.eval_interval) * args.eval_iters * args.multi_task_ratio
            multi_val_data.batch_sampler.start_iter = start_iter_val % len(multi_val_data)
    if train_data is not None:
        train_data_iterator = iter(train_data)
    else:
        train_data_iterator = None
    if multi_train_data is not None:
        multi_train_iterator = iter(multi_train_data)
    else:
        multi_train_iterator = None
    if val_data is not None:
        val_data_iterator = iter(val_data)
    else:
        val_data_iterator = None
    if multi_val_data is not None:
        multi_val_iterator = iter(multi_val_data)
    else:
        multi_val_iterator = None

    iteration = 0
    if args.train_iters > 0:
        # if args.do_train:
            # stack.callback(save_on_exit, args, model, optimizer, lr_scheduler)
        iteration, skipped = train(model, optimizer,
                                    lr_scheduler,
                                    (train_data_iterator, multi_train_iterator),
                                    (val_data_iterator, multi_val_iterator),
                                    timers, args, summary_writer=summary_writer,
                                    hooks={"forward_step": forward_step})

        # if args.do_valid:
        #     prefix = 'the end of training for val data'
        #     evaluate_and_print_results(prefix, (val_data_iterator, multi_val_iterator),
        #                                model, args, timers, verbose=False, forward_step_func=forward_step)

    if args.save and iteration != 0:
        save_checkpoint(iteration, model, optimizer, lr_scheduler, args)

    # if test_data is not None:
    #     test_data_iterator = iter(test_data)
    # else:
    #     test_data_iterator = None

    # if args.do_test:
    #     # Run on test data.
    #     prefix = 'the end of training for test data'
    #     evaluate_and_print_results(prefix, (test_data_iterator, None),
    #                                model, args, timers, verbose=True, forward_step_func=forward_step)


if __name__ == "__main__":
    # Arguments.
    py_parser = argparse.ArgumentParser(add_help=False)
    T5Model.add_model_specific_args(py_parser)
    known, args_list = py_parser.parse_known_args()
    args = get_args(args_list)
    args = argparse.Namespace(**vars(args), **vars(known))
    convert_hf_to_ds('/'.join(args.load.split('/')[:-1]),args.load.split('/')[-1])
    main(args)
