#############################################################################
# Attention! Only for evaluation now, not test for train yet! But I think you can change just a few codes so that it can work for train well.
#############################################################################

import os
import time
import torch
import math
import logging
import pathlib
import deepspeed
import torch.nn.functional as F
from torch.optim import Adam

from tqdm import tqdm
from torch.utils.data import DataLoader,Dataset,RandomSampler,SequentialSampler
from filelock import FileLock
from transformers import T5Tokenizer
from transformers import AdamW, Adafactor, get_linear_schedule_with_warmup
from transformers.optimization import AdafactorSchedule
from torch.optim import Optimizer

import datasets
from configure_data import configure_data, build_multi_task_dataset, make_tokenizer

from SwissArmyTransformer.training.deepspeed_training import initialize_distributed, \
    set_random_seed, setup_model_and_optimizer, get_model, get_optimizer_param_groups
from SwissArmyTransformer import mpu
from SwissArmyTransformer.model import T5Model
from SwissArmyTransformer.training.deepspeed_training import initialize_distributed, \
    set_random_seed, get_model, get_optimizer_param_groups, setup_model_and_optimizer

from SwissArmyTransformer.generation.autoregressive_sampling import filling_sequence, update_mems
from SwissArmyTransformer.generation.sampling_strategies import BeamSearchStrategy, BaseStrategy
from SwissArmyTransformer.model.mixins import CachedAutoregressiveMixin
from SwissArmyTransformer.training.utils import print_rank_0

import data_utils
from utils import get_checkpoint_iteration, get_checkpoint_name, load_checkpoint, print_and_save_args
logger = logging.getLogger(__name__)

from multiprocessing import Pool
# class DictDataset(Dataset):
# 	"""A dataset of tensors that uses a dictionary for key-value mappings"""

# 	def __init__(self, **tensors):
# 		tensors.values()

# 		assert all(next(iter(tensors.values())).size(0) == tensor.size(0) for tensor in tensors.values())
# 		self.tensors = tensors

# 	def __getitem__(self, index):
# 		return {key: tensor[index] for key, tensor in self.tensors.items()}

# 	def __len__(self):
# 		return next(iter(self.tensors.values())).size(0)

def build_model_and_optimizer(args,checkpoint_path):
    model=get_model(args,T5Model)
    model.decoder.add_mixin('auto-regressive', CachedAutoregressiveMixin())
    model.disable_untrainable_params()
    optimizer=None;lr_scheduler=None
    if args.do_train:
        param_groups=get_optimizer_param_groups(model)
        if args.optimizer.lower()=='adamw':
            optimizer=AdamW(param_groups,lr=args.lr,eps=args.adam_eps)
            lr_scheduler=get_linear_schedule_with_warmup(optimizer,num_warmup_steps=int(args.num_training_steps*args.warmup),num_training_steps=args.num_training_steps)
        elif args.optimizer.lower()=='adafactor':
            # optimizer=Adafactor(param_groups,lr=args.lr,relative_step = False)
            optimizer = Adafactor(param_groups, scale_parameter=False, relative_step=False, warmup_init=False, lr=args.lr)
            lr_scheduler = AdafactorSchedule(optimizer, initial_lr=args.lr) if optimizer is not None else None
        elif args.optimizer.lower()=='adam':
            optimizer=Adam(param_groups,lr=args.lr)
            lr_scheduler=get_linear_schedule_with_warmup(optimizer,num_warmup_steps=int(args.num_training_steps*args.warmup),num_training_steps=args.num_training_steps)
        if args.deepspeed:
            print_rank_0("DeepSpeed is enabled.")
            model,optimizer,_,lr_scheduler=deepspeed.initialize(
                model=model,
                model_parameters=param_groups,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                args=args,
                mpu=mpu,
                dist_init_required=False,
            )
        else: 
            raise NotImplementedError('Currently, we only support training with deepspeed.')

        if args.load is not None:
            with FileLock(os.path.join(pathlib.Path.home(), "checkpoint_lock"), timeout=-1):
                args.iteration = load_checkpoint(model, optimizer, lr_scheduler, args, no_deepspeed=args.no_deepspeed_load)
            # if args.no_load_lr_scheduler:
            # 	lr_scheduler.num_iters = args.iteration
        else:
            args.iteration = 0
    if args.do_train==False and args.load is not None:
        module=model.module if hasattr(model,"module") else model
        with FileLock(os.path.join(pathlib.Path.home(), "checkpoint_lock"), timeout=-1):
            load_pretrained(args,module,checkpoint_path)

    torch.distributed.barrier()
    # if torch.distributed.get_rank()==0:
    # 	print_and_save_args(args,verbose=True,log_dir=args.save)
    return model,optimizer,lr_scheduler

def load_pretrained(args, model, checkpoint_path):
    load_dir, tag, release, success = get_checkpoint_iteration(checkpoint_path)
    checkpoint_name = get_checkpoint_name(load_dir, tag, release)
    if mpu.get_data_parallel_rank() == 0:
        print('global rank {} is loading pretrained model {}'.format(torch.distributed.get_rank(), checkpoint_name))
    # Load the checkpoint.
    sd = torch.load(checkpoint_name, map_location='cpu')
    # Model.
    if args.block_lm and args.old_checkpoint:
        sd['module']['transformer.word_embeddings.weight'] = sd['module']['word_embeddings.weight']
        del sd['module']['word_embeddings.weight']
        sd['module']['mixins.block_position_embedding.block_position_embeddings.weight'] = sd['module'][
            'transformer.block_position_embeddings.weight']
        del sd['module']['transformer.block_position_embeddings.weight']

    missing_keys, unexpected_keys = model.load_state_dict(sd['module'], strict=False)
    if missing_keys or unexpected_keys:
        print_rank_0(f"Missing keys {missing_keys}, unexpected keys {unexpected_keys}")

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    # This function has been mostly taken from huggingface conversational ai code at
    # https://medium.com/huggingface/how-to-build-a-state-of-the-art-conversational-ai-with-transfer-learning-2d818ac26313

    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # convert to 1D
        # logits = logits.view(logits.size()[1]).contiguous()
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        # indices_to_remove = sorted_indices[sorted_indices_to_remove]
        indices_to_remove = sorted_indices_to_remove.scatter(1,sorted_indices,sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
        # going back to 2D
        # logits = logits.view(1, -1).contiguous()

    return logits

def _get_ngrams(ngram_size: int, prev_input_ids: torch.Tensor, num_hypos: int):
    generated_ngrams = [{} for _ in range(num_hypos)]
    for idx in range(num_hypos):
        gen_tokens = prev_input_ids[idx].tolist()
        generated_ngram = generated_ngrams[idx]
        for ngram in zip(*[gen_tokens[i:] for i in range(ngram_size)]):
            prev_ngram_tuple = tuple(ngram[:-1])
            generated_ngram[prev_ngram_tuple] = generated_ngram.get(prev_ngram_tuple, []) + [ngram[-1]]
    return generated_ngrams

def _get_generated_ngrams(banned_ngrams, prev_input_ids, ngram_size, cur_len):
    # Before decoding the next token, prevent decoding of ngrams that have already appeared
    start_idx = cur_len + 1 - ngram_size
    ngram_idx = tuple(prev_input_ids[start_idx:cur_len].tolist())
    return banned_ngrams.get(ngram_idx, [])

def norepeat_ngram_logits(input_ids: torch.LongTensor,scores: torch.FloatTensor, ngram_size: int=5) ->torch.FloatTensor:
    def _calc_banned_ngram_tokens(ngram_size,prev_input_ids,num_hypos,cur_len):
        if cur_len+1<ngram_size:
            return [[] for _ in range(num_hypos)]
        generated_ngrams=_get_ngrams(ngram_size,prev_input_ids,num_hypos)
        # print('generated_ngrams',generated_ngrams)
        banned_tokens=[_get_generated_ngrams(generated_ngrams[hypo_idx],prev_input_ids[hypo_idx],ngram_size,cur_len) for hypo_idx in range(num_hypos)]
        # print('banned_tokens',banned_tokens)
        return banned_tokens
    num_batch_hypotheses=scores.shape[0]
    cur_len=input_ids.shape[-1]
    banned_batch_tokens=_calc_banned_ngram_tokens(ngram_size,input_ids,num_batch_hypotheses,cur_len)
    for i,banned_tokens in enumerate(banned_batch_tokens):
        # print(scores.shape,i,banned_tokens)
        scores[i,banned_tokens]=-math.inf
    return scores

def decode_ids(tokenizer,ids):
    return [tokenizer.decode(id) for id in ids]

class CollatorForT0Data:
    def __init__(self,tokenizer,multi_src_seq_length,multi_tgt_seq_length):
        self.tokenizer=tokenizer
        self.multi_src_seq_length=multi_src_seq_length
        self.multi_tgt_seq_length=multi_tgt_seq_length
        self.end_token_id=tokenizer.get_command('eop').Id

    def __call__(self,examples):
        # examples: {inputs, targets}, possible:{task_names,prompt_names,input_prompt_mask_poss}
        input_ids=[];attention_mask=[];labels=[]
        # print(examples)
        for e in examples:
            # print(e['input_tokens'])
            x=self.tokenizer.EncodeAsIds(e['input_tokens']).__dict__['tokenization']
            input_ids.append(x[:self.multi_src_seq_length-1]+[self.end_token_id])
            
            if 'label' in e:
                la=self.tokenizer.EncodeAsIds(e['label']).__dict__['tokenization']
                labels.append(la[:self.multi_tgt_seq_length-1]+[self.end_token_id])

        max_input_len=min(self.multi_src_seq_length,max([len(x) for x in input_ids]))
        attention_mask=[[1]*len(x)+[0]*(max_input_len-len(x)) for x in input_ids]
        input_ids=[x[:max_input_len]+[0]*(max_input_len-len(x)) for x in input_ids]

        if 'label' in examples[0]:
            max_label_len=min(self.multi_tgt_seq_length,max([len(x) for x in labels]))
            labels=[label[:max_label_len]+[-100]*(max_label_len-len(label)) for label in labels]
            data={'input_ids':input_ids,'attention_mask':attention_mask,'labels':labels}
        else:
            data={'input_ids':input_ids,'attention_mask':attention_mask}
        if 'label_ids' in examples[0]:
            data['label_ids']=[e['label_ids'] for e in examples]
        ########################### do not apply tokenization ########################
        # data = {k: [e[k] for e in examples] for k in examples[0].keys()}
        ##############################################################################
        return data   

class TransformerModelWrapper:
    def __init__(self,args,checkpoint_path):
        self.tokenizer=make_tokenizer(args)
        args.decoder_start_token_id = self.tokenizer.get_command('sop').Id
        args.end_token_id=self.tokenizer.get_command('eop').Id
        args.pad_token_id=0
        self.args=args
        # self.model,self.optimizer=setup_model_and_optimizer(args,model_cls=T5Model)
        # load_pretrained(self.model.module, checkpoint_path, args, optimizer=self.optimizer)
        # args.iteration = load_checkpoint(self.model, self.optimizer, lr_scheduler, args, no_deepspeed=args.no_deepspeed_load)
        # self.tokenizer=T5Tokenizer.from_pretrained(os.path.join('../../huggingface_models',args.model_name_or_path)) if tokenizer is None else tokenizer
        self.model,self.optimizer,self.lr_scheduler=build_model_and_optimizer(args,checkpoint_path) # we do not build optimizer if args.do_train==False
        self.world_size=args.world_size
        # if args.eval_batch_size is not None:
        #     self.eval_batch_size=args.eval_batch_size*self.world_size
        # else:
        #     self.eval_batch_size=args.batch_size*self.world_size
        # self.decoder_start_token_id=self.tokenizer.pad_token_id

    def _make_data_loader(self,dataset,batch_size,shuffle=False,world_size=None):
        world_size = torch.distributed.get_world_size(group=mpu.get_data_parallel_group())
        args=self.args
        # if world_size is None:
        # 	world_size=self.world_size
        rank = torch.distributed.get_rank(group=mpu.get_data_parallel_group())
        if args.loader_scatter is not None:
            loader_scatter = min(args.loader_scatter, mpu.get_data_parallel_world_size())
            rank = rank // loader_scatter
            world_size = world_size // loader_scatter
            batch_size = batch_size // loader_scatter
        distributed = world_size > 1
        if shuffle==True:
            # sampler=torch.utils.data.RandomSampler(dataset)
            print('batch_size:',batch_size,'train_iters:',args.train_iters,'accu:',args.gradient_accumulation_steps)
            sampler = data_utils.samplers.RandomSampler(dataset,replacement=True,num_samples=batch_size*args.train_iters*args.gradient_accumulation_steps)
            # sampler = data_utils.samplers.RandomSampler(dataset,replacement=False)
        else:
            sampler = torch.utils.data.SequentialSampler(dataset)
        drop_last = distributed
        # the GPUs in the same model parallel group receive the same data
        if distributed:
            batch_sampler = data_utils.samplers.DistributedBatchSampler(sampler, batch_size, drop_last, rank,
                            world_size,gradient_accumulation_steps=args.gradient_accumulation_steps)
        else:
            batch_sampler = torch.utils.data.BatchSampler(sampler,batch_size,drop_last)
        collate_fn=CollatorForT0Data(self.tokenizer,self.args.multi_src_seq_length,self.args.multi_tgt_seq_length)
        data_loader = torch.utils.data.DataLoader(dataset,
                            batch_sampler=batch_sampler,
                            num_workers=args.num_workers,
                            pin_memory=True,
                            collate_fn=collate_fn)
        return data_loader        

    def build_rank_sampler(self,dataset,collator='t0',batch_size=-1):
        # from finetune_self_train_t5_20220217 import CollatorForIrregularData
        sampler = torch.utils.data.SequentialSampler(dataset)
        collate_fn=CollatorForT0Data(self.tokenizer,self.args.multi_src_seq_length,self.args.multi_tgt_seq_length)
        batch_size=self.args.eval_batch_size if batch_size==-1 else batch_size
        data_loader = torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=False,sampler=sampler,collate_fn=collate_fn)
        # print('len(dataset)',len(dataset),'len(data_loader)',len(data_loader))
        return data_loader


    def del_finetuned_model(self):
        self.model.cpu()
        self.model = None
        torch.cuda.empty_cache()

    def _shift_right(self,input_ids):
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = self.args.decoder_start_token_id
        return shifted_input_ids

    def get_eval_batch(self,data):
        args=self.args
        datatype=torch.int64
        batch={k:torch.tensor(v).to(args.device) for k,v in data.items()}
        batch['attention_mask']=batch['attention_mask'].unsqueeze(1).unsqueeze(2)
        keys=batch.keys()
        batch=mpu.broadcast_data(keys,batch,datatype)
        if args.fp16:
            batch['attention_mask']=batch['attention_mask'].half()
        elif args.bf16:
            batch['attention_mask']=batch['attention_mask'].bfloat16()
        return batch
    
    def eval_cls_step(self,data_iterator,do_collect=True,**kwargs):
        self.model.eval()
        data=next(data_iterator)
        batch=self.get_eval_batch(data)
        input_ids,attention_mask,labels=batch['input_ids'],batch['attention_mask'],batch['labels']
        # feature=model.encode(input_ids,attention_mask)
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100,reduce=False)
        with torch.no_grad():
            decoder_input_ids=self._shift_right(labels)
            outputs=self.model(enc_input_ids=input_ids,enc_attention_mask=attention_mask,dec_input_ids=decoder_input_ids)
            lm_logits=outputs[1]
            scores = -loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1)).view(labels.shape[0],-1)
            scores = scores.sum(dim=1)/torch.sum(labels!=-100,dim=1)

        if self.args.world_size>1 and do_collect==True:
            output_list=[scores.clone() for _ in range(self.args.world_size)]
            torch.distributed.all_gather(output_list,scores)
            return torch.cat(output_list,dim=0).detach().cpu().float().numpy().tolist()
        else:
            return scores.detach().cpu().float().numpy().tolist()

    def constrained_eval(self,inputs,choices,**kwargs):
        args=self.args
        input_tokens=[];labels=[]
        for input_token,choice in zip(inputs,choices):
            for label in choice:
                input_tokens.append(input_token)
                labels.append(label)
        eval_dataset=datasets.Dataset.from_dict({'input_tokens':input_tokens,'label':labels})
        eval_dataloader=self._make_data_loader(eval_dataset,args.eval_batch_size*args.world_size)
        eval_data_iterator=iter(eval_dataloader)
        total_logits=[]
        for _ in tqdm(range(len(eval_data_iterator)),disable=(args.rank!=0)):
            if _%10==0: print_rank_0('eval_step:{}/{}'.format(_,len(eval_data_iterator)))
            logits=self.eval_cls_step(eval_data_iterator,**kwargs)
            total_logits+=logits

        remain_length=len(eval_dataset)%(args.eval_batch_size*args.world_size)
        remain_start_idx=len(eval_dataset)-remain_length
        if args.world_size>1 and remain_start_idx!=len(eval_dataset):
            remain_gen_dataset=eval_dataset.select(range(remain_start_idx,len(eval_dataset)))
            new_eval_batch_size=remain_length//args.world_size
            print('remain_start_idx, new_eval_batch_size, total_len',remain_start_idx,new_eval_batch_size,len(eval_dataset))
            eval_dataloader=self._make_data_loader(remain_gen_dataset,new_eval_batch_size*args.world_size)
            eval_data_iterator=iter(eval_dataloader)
            for _ in tqdm(range(len(eval_data_iterator)),disable=(args.rank!=0)):
                logits=self.eval_cls_step(eval_data_iterator,**kwargs)
                total_logits+=logits

            remain_start_idx=remain_start_idx+new_eval_batch_size*args.world_size
            if remain_start_idx!=len(eval_dataset):
                print('remain_start_idx2',remain_start_idx)
                remain_gen_dataset=eval_dataset.select(range(remain_start_idx,len(eval_dataset)))
                # print('iny',len(remain_cls_dataset))
                # remain_cross_data=make_data_loader(remain_cls_dataset,tokenizer,args.eval_batch_size,math.ceil(args.cross_train_iters*args.cross_batch_size/args.eval_batch_size),args,shuffle=False,collator='t0')
                remain_cross_data=self.build_rank_sampler(remain_gen_dataset)
                remain_cross_data_iterator=iter(remain_cross_data)
                for _ in range(len(remain_cross_data_iterator)):
                    logits=self.eval_cls_step(remain_cross_data_iterator,do_collect=False,**kwargs)
                    total_logits+=logits
        # import pdb 
        # pdb.set_trace()
        rearanged_logits=[]
        cnt=0
        for input_token,choice in zip(inputs,choices):
            rearanged_logits.append([])
            for _ in range(len(choice)):
                rearanged_logits[-1].append(total_logits[cnt])
                cnt+=1
        return rearanged_logits

    def generate_sequence(self,enc_input_ids,enc_attention_mask=None,output_num_per_sample=1,gen_type='greedy',**kwargs):
        # greedy, support for batch_size>1
        # if gen_type=='greedy': output_num_per_sample=1
        assert (gen_type=='greedy' and output_num_per_sample==1) or gen_type=='sample'
        args=self.args
        # gen_type='greedy'
        self.model.eval()
        module=self.model.module if hasattr(self.model,"module") else self.model
        cur_len=0;max_length=args.multi_tgt_seq_length;max_memory_length=100000
        batch_size=enc_input_ids.shape[0]
        with torch.no_grad():
            # enc_attention_mask=enc_attention_mask.unsqueeze(1).unsqueeze(2)
            encoder_outputs=module.encode(enc_input_ids,enc_attention_mask)
            # print(encoder_outputs.shape)
            if output_num_per_sample>1:
                encoder_outputs=encoder_outputs.unsqueeze(1).expand(encoder_outputs.shape[0],output_num_per_sample,encoder_outputs.shape[-2],encoder_outputs.shape[-1]).reshape(-1,encoder_outputs.shape[-2],encoder_outputs.shape[-1])
                enc_attention_shape=enc_attention_mask.shape
                enc_attention_mask=enc_attention_mask.unsqueeze(1).expand([enc_attention_mask.shape[0],output_num_per_sample]+list(enc_attention_shape[1:])).reshape([-1]+list(enc_attention_shape)[1:])
                batch_size=batch_size*output_num_per_sample
            done = [False] * batch_size
            ret_logits=torch.zeros(batch_size,dtype=encoder_outputs.dtype)
            tokens=torch.ones((batch_size,1),dtype=torch.long).to(args.device)*args.decoder_start_token_id
            attention_mask=torch.ones((batch_size,1,max_length,max_length),device=args.device,dtype=torch.long)
            position_ids=torch.zeros(1, max_length, device=args.device, dtype=torch.long)
            index=0;counter=0;mems=None
            while(counter<max_length-1):
                logits,*output_per_layers=module.decoder(
                    tokens[:, -1:],
                    position_ids[...,index:counter+1],
                    attention_mask[...,index:counter+1,:counter+1],
                    mems=mems,
                    encoder_outputs=encoder_outputs,
                    cross_attention_mask=enc_attention_mask)
                mem_kv = [o['mem_kv'] for o in output_per_layers]
                mems = update_mems(mem_kv, mems, max_memory_length=max_memory_length)
                # print('sesssssssssssssssssssssssssssss')
                next_token_logits=logits.view(logits.shape[0]*logits.shape[1],-1)
                if 'norepeat_ngram_size' in kwargs and kwargs['norepeat_ngram_size']!=-1:
                    next_token_logits=norepeat_ngram_logits(tokens,next_token_logits,kwargs['norepeat_ngram_size'])
                next_token_logits=top_k_top_p_filtering(next_token_logits,top_k=args.top_k,top_p=args.top_p)
                # import pdb 
                # pdb.set_trace()
                next_token_logits=next_token_logits/args.temperature
                log_probs=F.softmax(next_token_logits,dim=-1)
                if gen_type=='sample':
                    next_tokens = torch.multinomial(log_probs,num_samples=1).squeeze(1)
                else:
                    next_tokens = torch.argmax(log_probs, dim=-1)
                for i, next_token in enumerate(next_tokens.view(-1).tolist()):
                    if next_token in [args.end_token_id,args.pad_token_id]:
                        done[i] = True
                    if done[i]:
                        if tokens[i,-1] not in [args.end_token_id,args.pad_token_id]:
                            next_tokens[i] = args.end_token_id
                        else:
                            next_tokens[i] = args.pad_token_id
                    if next_tokens[i]!=args.pad_token_id:
                        ret_logits[i]+=log_probs[i,next_tokens[i]].item()
                # print(tokens.shape,next_tokens)
                tokens=torch.cat([tokens,next_tokens.unsqueeze(1)],dim=-1)
                if all(done):
                    break
                counter += 1
                index = counter
        ret_tokens=torch.ones((batch_size,max_length),dtype=torch.long)*args.pad_token_id
        ret_tokens[...,:tokens.shape[-1]-1]=tokens[...,1:]
        ret_logits=ret_logits/(torch.sum((ret_tokens!=0),dim=1))
        # import pdb 
        # pdb.set_trace()
        return ret_tokens.view(enc_input_ids.shape[0],output_num_per_sample,-1),ret_logits.view(enc_input_ids.shape[0],output_num_per_sample)

    def eval_gen_step(self,data_iterator,do_collect=True,**kwargs):
        args=self.args
        data=next(data_iterator)
        # model.eval()
        batch=self.get_eval_batch(data)
        input_ids,attention_mask=batch['input_ids'],batch['attention_mask']
        if 'output_num_per_sample' not in kwargs: kwargs['output_num_per_sample']=1
        if 'gen_type' not in kwargs: kwargs['gen_type']='greedy'
        output,logits=self.generate_sequence(enc_input_ids=input_ids,enc_attention_mask=attention_mask,**kwargs)
        
        if args.world_size>1 and do_collect==True: 
            output=output.to(input_ids.device)
            logits=logits.to(input_ids.device)       
            output_list=[output.clone() for _ in range(args.world_size)]
            torch.distributed.all_gather(output_list,output)
            output=torch.cat(output_list,dim=0).detach().cpu().int().numpy().tolist()
            logits_list=[logits.clone() for _ in range(args.world_size)]
            torch.distributed.all_gather(logits_list,logits)
            logits=torch.cat(logits_list,dim=0).detach().cpu().float().numpy().tolist()
        else:
            output=output.detach().cpu().int().numpy().tolist()
            logits=logits.detach().cpu().float().numpy().tolist()
        return output,logits        

    def direct_eval(self,eval_dataset,**kwargs):
        args=self.args
        eval_dataloader=self._make_data_loader(eval_dataset,args.eval_batch_size*args.world_size)
        eval_data_iterator=iter(eval_dataloader)
        total_outputs=[];total_logits=[]
        for _ in tqdm(range(len(eval_data_iterator)),disable=(args.rank!=0)):
            if _%10==0: print_rank_0('eval_step:{}/{}'.format(_,len(eval_data_iterator)))
            label_ids,logits=self.eval_gen_step(eval_data_iterator,**kwargs)
            seq=[[self.tokenizer.decode(id) for id in ids] for ids in label_ids]
            total_outputs+=seq
            total_logits+=logits
        # import pdb 
        # pdb.set_trace()
        remain_length=len(eval_dataset)%(args.eval_batch_size*args.world_size)
        remain_start_idx=len(eval_dataset)-remain_length
        # print(len(cls_dataset),len(cls_dataset.expanded_dataset['input_tokens']),args.eval_batch_size,remain_start_idx)
        if args.world_size>1 and remain_start_idx!=len(eval_dataset):
            remain_gen_dataset=eval_dataset.select(range(remain_start_idx,len(eval_dataset)))
            new_eval_batch_size=remain_length//args.world_size
            if new_eval_batch_size!=0:
                print('remain_start_idx, new_eval_batch_size, total_len',remain_start_idx,new_eval_batch_size,len(eval_dataset))
                eval_dataloader=self._make_data_loader(remain_gen_dataset,new_eval_batch_size*args.world_size)
                eval_data_iterator=iter(eval_dataloader)
                for _ in tqdm(range(len(eval_data_iterator)),disable=(args.rank!=0)):
                    label_ids,logits=self.eval_gen_step(eval_data_iterator,**kwargs)
                    seq=[[self.tokenizer.decode(id) for id in ids] for ids in label_ids]
                    total_outputs+=seq
                    total_logits+=logits

            remain_start_idx=remain_start_idx+new_eval_batch_size*args.world_size
            if remain_start_idx!=len(eval_dataset):
                print('remain_start_idx2',remain_start_idx)
                remain_gen_dataset=eval_dataset.select(range(remain_start_idx,len(eval_dataset)))
                # print('iny',len(remain_cls_dataset))
                # remain_cross_data=make_data_loader(remain_cls_dataset,tokenizer,args.eval_batch_size,math.ceil(args.cross_train_iters*args.cross_batch_size/args.eval_batch_size),args,shuffle=False,collator='t0')
                remain_cross_data=self.build_rank_sampler(remain_gen_dataset)
                remain_cross_data_iterator=iter(remain_cross_data)
                for _ in range(len(remain_cross_data_iterator)):
                    label_ids,logits=self.eval_gen_step(remain_cross_data_iterator,do_collect=False,**kwargs)
                    seq=[[self.tokenizer.decode(id) for id in ids] for ids in label_ids]
                    total_outputs+=seq
                    total_logits+=logits
        return total_outputs,total_logits

    def eval(self,inputs,choices=None, **kwargs):
        # inputs are a list of examples, we will evaluate with batch
        if choices is not None:
            return self.constrained_eval(inputs,choices)
        else:
            eval_dataset=datasets.Dataset.from_dict({'input_tokens':inputs})
            return self.direct_eval(eval_dataset,**kwargs)                

    def train_step(self, input_ids, attention_mask, labels):
        self.model.train()
        loss_fct=torch.nn.CrossEntropyLoss(ignore_index=-100)
        loss_mask=(labels!=-100).detach()
        decoder_input_ids=self._shift_right(labels)
        outputs=self.model(enc_input_ids=input_ids,enc_attention_mask=attention_mask,dec_input_ids=decoder_input_ids)
        lm_logits=outputs[1]
        # import pdb 
        # pdb.set_trace()
        loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
        return loss




if __name__ == '__main__':
    from SwissArmyTransformer.training.deepspeed_training import initialize_distributed, set_random_seed
    # from arguments import get_args
    # args,_=get_args()
    import argparse
    from arguments import get_args
    py_parser = argparse.ArgumentParser(add_help=False)
    T5Model.add_model_specific_args(py_parser)
    known, args_list = py_parser.parse_known_args()
    args,_ = get_args(args_list)
    args = argparse.Namespace(**vars(args), **vars(known))
    args.model_name_or_path='t5-large-lm-adapt'

    torch.backends.cudnn.enabled = False
    initialize_distributed(args)
    set_random_seed(args.seed)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.rank in [-1, 0] else logging.WARN,
    )
    if args.test_ckpt is None:
        T5wrapper=TransformerModelWrapper(args,args.load)
        args.save="results/{}"
        if os.path.exists(args.save)==False:
            os.makedirs(args.save)
    else:
        T5wrapper=TransformerModelWrapper(args,args.test_ckpt)
        args.save=args.test_ckpt
    import pdb 
    pdb.set_trace()
    text1='Paraphrase the following sentence:  What term in biotechnology means a genetically exact copy of an organism?'
    text2='Summarise the article: \n\nThe perfect murder is foiled when a wife(played by Mary Ellen Trainor, once the wife to director Robert Zemeckis, who helmed this episode), who murders her husband with a poker, has the misfortune of receiving a visitor as she is about to move the body outside..an escaped insane madman dressed in a Santa Claus suit(played by a deviously hideous Larry Drake). She fends for her life while trying to find a way of hiding her husband\'s corpse. She decides to use an ax, once she downs the Santa killer who misses several chances to chop off the woman\'s head, to frame the killer for her husband\'s murder. Santa killer locks her in a closet and pursues the woman\'s daughter as she tries desperate to free herself to save the child.<br /><br />This episode of TALES FROM THE CRYPT just recycles tired material involving the old "Santa kills" theme while also adding the oft-used(add nauseum)woman-murders-her-husband-for-a-man-she\'s-been-cheating-with routine. It\'s essentially Trainor trying to find a way to avoid being caught with a dead body she kills while also keeping a safe distance from a maniac. There\'s nothing refreshing or new about this plot which pretty much goes through the motions. Not one of the show\'s highlights.'
    print(T5wrapper.eval([text1,text2]))
    text=text1
    print(T5wrapper.eval([text]))
    import pdb 
    pdb.set_trace()
    # result=T5wrapper.eval(["I study in "],[['American','China']])
    # result=T5wrapper.eval(["I study in "])
    # import pdb 
    # pdb.set_trace()
    # T5wrapper.model()




'''
NCCL_DEBUG=info NCCL_IB_DISABLE=0 NCCL_NET_GDR_LEVEL=2 CUDA_VISIBLE_DEVICES=7 deepspeed --master_port 32345 wrapper_t5.py \
--load /dataset/fd5061f6/yanan/huggingface_models/t5-large-lm-adapt \
--task_group_name default_T0_test \
--valid_isolate_type all --max_valid_num_per_dataset -1 --from_p3 \
--eval_type constraint \
--metric_funcs acc f1 \
--eval-batch-size 16 \
--multi-src-seq-length 1024 \
--multi-tgt-seq-length 256 \
--test-ckpt ../T0-Multi-Task_zj-adapt/checkpoints/mt-t5-lm-large-T0_adam_0.0001_default-T0-tasks_trisoall_validisorand_fold0of2/t5-large-lm-adapt/10000 \
--deepspeed \
--deepspeed_config "config/t5_lm/config_t5_lm_large.json" \
--save test \
--no-deepspeed-load \
--no-load-optim \
--no-load-lr-scheduler \
--no-load-rng \
--no-load-iteration \
--bert-mask-ratio 0.15 \
--avg-block-length 3 \
--experiment-name test \
--model-parallel-size 1 \
--t5-model \
--vocab-size 32128 \
--num-layers 24 \
--hidden-size 1024 \
--inner-hidden-size 2816 \
--num-attention-heads 16 \
--hidden-size-per-attention-head 64 \
--relative-attention-num-buckets 32 \
--no-share-embeddings \
--gated-gelu-mlp \
--layernorm-epsilon 1e-6 \
--init-method-std 1.0 \
--seq-length 512 \
--shuffle \

'''