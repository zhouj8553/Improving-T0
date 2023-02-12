#! /bin/bash
task_set=$2

MASTER_PORT=$(shuf -n 1 -i 10000-65535)

source $1

DATESTR=$(date +"%m-%d-%H-%M")

mkdir logs
NCCL_DEBUG=info NCCL_IB_DISABLE=0 NCCL_NET_GDR_LEVEL=2 deepspeed --master_port 12345 pretrain_t5.py \
 ${total_options}
wait
set +x

