#! /bin/bash

# Change for multinode config

task_set=$2

NUM_WORKERS=$3
NUM_GPUS_PER_WORKER=$4
MP_SIZE=1
MASTER_PORT=$(shuf -n 1 -i 10000-65535)

source $1

HOST_FILE_PATH=hostfiles/hostfile
DATESTR=$(date +"%m-%d-%H-%M")

mkdir logs
run_cmd="${OPTIONS_NCCL} deepspeed --master_port ${MASTER_PORT} --num_nodes ${NUM_WORKERS} --num_gpus
${NUM_GPUS_PER_WORKER} --hostfile ${HOST_FILE_PATH} pretrain_t5.py --model-parallel-size ${MP_SIZE} ${total_options} 2>&1 | tee logs/log-${DATESTR}.txt"
echo ${run_cmd}
eval ${run_cmd}

wait
set +x

