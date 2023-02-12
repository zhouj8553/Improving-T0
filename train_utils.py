import math

import deepspeed
import torch.cuda
import torch.distributed

from SwissArmyTransformer import mpu
from SwissArmyTransformer.training.deepspeed_training import train_step
from SwissArmyTransformer.training.utils import report_memory
from configure_data import configure_data
from utils import print_rank_0, save_checkpoint


def evaluate_and_print_results(prefix, data_iterator, model,
                               args, timers, forward_step_func, verbose=False, step=None, summary_writer=None):
    """Helper function to evaluate and dump results on screen."""
    lm_loss, gpt_loss, bert_loss, sent_loss, multi_loss = evaluate(data_iterator, model, args, timers, verbose=verbose,
                                                                   forward_step_func=forward_step_func)

    lm_ppl = math.exp(min(20, lm_loss))
    report_evaluate_metrics(summary_writer, prefix, lm_loss, lm_ppl, gpt_loss, bert_loss, sent_loss, multi_loss, step)

    return lm_loss


def evaluate(data_iterator, model, args, timers, forward_step_func, verbose=False, **kwargs):
    """Evaluation."""
    # Turn on evaluation mode which disables dropout.
    model.eval()

    total_lm_loss, total_gpt_loss, total_bert_loss, total_sent_loss, total_multi_loss = 0, 0, 0, 0, 0
    gpt_iters, bert_iters, sent_iters, multi_iters = 0, 0, 0, 0
    with torch.no_grad():
        iteration = 0
        while iteration < args.eval_iters:
            iteration += 1
            if verbose and iteration % args.log_interval == 0:
                print_rank_0('Evaluating iter {}/{}'.format(iteration, args.eval_iters))
            # Forward evaluation.
            lm_loss, mode = forward_step_func(data_iterator, model, args, timers, **kwargs)

            '''when contiguous memory optimizations are enabled, the buffers
            allocated by the optimizations are deallocated during backward pass
            in the absence of backward pass the buffers should be reset after each
            forward pass'''
            if args.deepspeed and args.deepspeed_activation_checkpointing:
                deepspeed.checkpointing.reset()

            lm_loss = lm_loss.data.detach().float().item()
            total_lm_loss += lm_loss
            mode = {name: value.item() for name, value in mode.items()}
            if mode['gpt'] != 0.0:
                total_gpt_loss += lm_loss
                gpt_iters += 1
            elif mode['bert'] != 0.0:
                total_bert_loss += lm_loss
                bert_iters += 1
            elif mode['sentence'] != 0.0:
                total_sent_loss += lm_loss
                sent_iters += 1
            elif mode['multi-task'] != 0.0:
                total_multi_loss += lm_loss
                multi_iters += 1
    # Move model back to the train mode.
    model.train()
    # Reduce across processes.
    loss_data = torch.cuda.FloatTensor(
        [total_lm_loss, total_gpt_loss, total_bert_loss, total_sent_loss, total_multi_loss, gpt_iters, bert_iters,
         sent_iters, multi_iters])
    torch.distributed.all_reduce(loss_data, group=mpu.get_data_parallel_group())
    loss_data = loss_data.tolist()
    total_lm_loss = loss_data[0] / args.eval_iters / (args.world_size / args.model_parallel_size)
    total_gpt_loss = loss_data[1] / loss_data[5] if loss_data[5] > 0 else 0
    total_bert_loss = loss_data[2] / loss_data[6] if loss_data[6] > 0 else 0
    total_sent_loss = loss_data[3] / loss_data[7] if loss_data[7] > 0 else 0
    total_multi_loss = loss_data[4] / loss_data[8] if loss_data[8] > 0 else 0
    return total_lm_loss, total_gpt_loss, total_bert_loss, total_sent_loss, total_multi_loss


def report_iteration_metrics(summary_writer, optimizer, lr, loss, elapsed_time, step, total_step, args):
    log_string = ' iteration {:8d}/{:8d} |'.format(step, total_step)
    log_string += ' elapsed time per iteration (ms): {:.1f} |'.format(elapsed_time)
    log_string += ' learning rate {:.3E} |'.format(lr)
    log_string += ' lm loss {:.6E} |'.format(loss)
    if args.fp16:
        log_string += ' loss scale {:.1f} |'.format(
            optimizer.cur_scale if args.deepspeed else optimizer.loss_scale)
    print_rank_0(log_string)
    if summary_writer is not None:
        summary_writer.add_scalar(f'Train/lr', lr, step)
        summary_writer.add_scalar(f'Train/train_loss', loss, step)
        summary_writer.add_scalar(f'Train/elapsed_time', elapsed_time, step)


def report_evaluate_metrics(summary_writer, prefix, loss, ppl, gpt_loss, bert_loss, sent_loss, multi_loss, step):
    string = ' validation loss at {}'.format(prefix)
    string += ' | LM loss: {:.6E}'.format(loss)
    string += ' | LM PPL: {:.6E}'.format(ppl)
    if gpt_loss != 0:
        string += ' | GPT loss: {:.6E}'.format(gpt_loss)
    if bert_loss != 0:
        string += ' | BERT loss: {:.6E}'.format(bert_loss)
    if sent_loss != 0:
        string += ' | Sent loss: {:.6E}'.format(sent_loss)
    if multi_loss != 0:
        string += ' | Multi loss: {:.6E}'.format(multi_loss)
    length = len(string) + 1
    print_rank_0('-' * 100)
    print_rank_0('-' * length)
    print_rank_0(string)
    print_rank_0('-' * length)
    if summary_writer is not None:
        summary_writer.add_scalar(f'Train/valid_ppl', ppl, step)
        summary_writer.add_scalar(f'Train/valid_loss', loss, step)
        if gpt_loss != 0:
            summary_writer.add_scalar(f'Train/valid_gpt_loss', gpt_loss, step)
        if bert_loss != 0:
            summary_writer.add_scalar(f'Train/valid_bert_loss', bert_loss, step)
        if sent_loss != 0:
            summary_writer.add_scalar(f'Train/valid_sent_loss', sent_loss, step)
        if multi_loss != 0:
            summary_writer.add_scalar(f'Train/valid_multi_loss', multi_loss, step)


def train(model, optimizer, lr_scheduler,
          train_data_iterator, val_data_iterator, timers, args, summary_writer=None, hooks={}):
    """Train the model."""

    forward_step = hooks['forward_step']
    # Turn on training mode which enables dropout.
    model.train()

    # Tracking loss.
    total_lm_loss = 0.0

    # Iterations.
    skipped_iters = 0

    timers('interval time').start()
    report_memory_flag = True
    while args.iteration < args.train_iters:

        lm_loss, skipped_iter, metrics = train_step(train_data_iterator,
                                                    model,
                                                    optimizer,
                                                    lr_scheduler,
                                                    args, timers, hooks=hooks)
        skipped_iters += skipped_iter
        args.iteration += 1

        # Update losses.
        total_lm_loss += lm_loss.data.detach().float()

        # Logging.
        if args.iteration % args.log_interval == 0:
            learning_rate = optimizer.param_groups[0]['lr']
            avg_lm_loss = total_lm_loss.item() / args.log_interval
            elapsed_time = timers('interval time').elapsed()
            report_iteration_metrics(summary_writer, optimizer, learning_rate, avg_lm_loss,
                                     elapsed_time * 1000.0 / args.log_interval, args.iteration, args.train_iters, args)
            total_lm_loss = 0.0
            if report_memory_flag:
                report_memory('after {} iterations'.format(args.iteration))
                report_memory_flag = False
            if args.deepspeed or args.DDP_impl == 'torch':
                timers.log(['forward', 'backward', 'optimizer',
                            'batch generator', 'data loader'],
                           normalizer=args.log_interval)
            else:
                timers.log(['forward', 'backward', 'allreduce', 'optimizer',
                            'batch generator', 'data loader'],
                           normalizer=args.log_interval)
        # Checkpointing
        if args.save and args.save_interval and args.iteration % args.save_interval == 0:
            save_checkpoint(args.iteration, model, optimizer, lr_scheduler, args)

        # Evaluation
        if args.eval_interval and args.iteration % args.eval_interval == 0 and args.do_valid:
            prefix = 'iteration {}'.format(args.iteration)
            evaluate_and_print_results(
                prefix, val_data_iterator, model, args, timers, verbose=False, step=args.iteration,
                summary_writer=summary_writer, forward_step_func=forward_step)

    return args.iteration, skipped_iters


def get_train_val_test_data(args, tokenizer):
    """Load the data on rank zero and boradcast number of tokens to all GPUS."""

    (train_data, val_data, test_data) = (None, None, None)
    # Data loader only on rank 0 of each model parallel group.
    if mpu.get_model_parallel_rank() == 0:
        data_config = configure_data()
        if args.transformer_xl:
            data_set_type = "GPT-XL"
        else:
            data_set_type = "Block"
        data_config.set_defaults(data_set_type=data_set_type, transpose=False)
        train_data, val_data, test_data = data_config.apply(args, tokenizer)

        data_counts = torch.cuda.LongTensor([int(args.do_train), int(args.do_valid), int(args.do_test)])
    else:
        data_counts = torch.cuda.LongTensor([0, 0, 0])

    # Broadcast num tokens.
    torch.distributed.broadcast(data_counts,
                                mpu.get_model_parallel_src_rank(),
                                group=mpu.get_model_parallel_group())
    args.do_train = data_counts[0].item()
    args.do_valid = data_counts[1].item()
    args.do_test = data_counts[2].item()

    return train_data, val_data, test_data