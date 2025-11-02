#!/bin/bash
# Test NCCL communication between GPUs

echo "Testing NCCL communication..."

export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0

# Simple test - run a distributed PyTorch command
torchrun --nproc_per_node=4 --master_port=29500 \
    python3 -c "
import torch
import torch.distributed as dist
import os

if 'LOCAL_RANK' in os.environ:
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    tensor = torch.ones(1).cuda() * rank
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    print(f'Rank {rank}: Result = {tensor.item()}')
    dist.destroy_process_group()
else:
    print('Not in distributed mode')
"

