#!/usr/bin/env python3
"""Simple NCCL test script for torchrun"""
import torch
import torch.distributed as dist
import os

if __name__ == "__main__":
    # Check if running in distributed mode
    if 'LOCAL_RANK' in os.environ or 'RANK' in os.environ:
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl')
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        print(f'Rank {rank}/{world_size}: Initialized on GPU {local_rank}')
        tensor = torch.ones(1).cuda() * rank
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        print(f'Rank {rank}: Result = {tensor.item()} (expected sum: {sum(range(world_size))})')
        dist.destroy_process_group()
    else:
        print('ERROR: Not running in distributed mode!')
        print('Please run this script with torchrun:')
        print('  torchrun --nproc_per_node=4 --master_port=29500 test_nccl_inline.py')
        print('Or use the test script:')
        print('  ./test_nccl.sh')

