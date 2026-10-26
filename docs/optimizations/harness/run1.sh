#!/bin/bash
# usage: run1.sh <gpus e.g. 0,1> <nproc> <port> <trace.json|/dev/null> <preset> <B> <L> [steps]
# env: XLLM_ROOT (repo), XEXT (xllm_extension build dir), CFG_<ModelConf field>=<value>, FSDP_RESHARD=0/1, FSDP_FP32_RS,
#      SAME_SEED=1 (same batch on all ranks), AWM_BACKEND / TN_BACKEND (default cute / cute), FSDP_SHARD_SIZE (HSDP), FSDP_PREFETCH_DEPTH, EOS_RATE, PIN_CPU
cd "$(dirname "$0")"
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-lo} NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1} AWM_BACKEND=${AWM_BACKEND:-cute} TN_BACKEND=${TN_BACKEND:-cute} CFG_sca_backend=${CFG_sca_backend:-cute}
gpus=$1; np=$2; port=$3; tr=$4; shift 4
CUDA_VISIBLE_DEVICES=$gpus MASTER_PORT=$port TRACE_OUT=$tr torchrun --nproc_per_node $np fsdp_trace.py "$@" 2>&1 | grep -v "Warning\|^USDT\|Python version"
