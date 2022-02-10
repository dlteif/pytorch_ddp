import numpy as np
import torch
import torch.nn as nn
from numpy import random
import random
import os
import time

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    if len(target.shape) > 1: return torch.tensor(1), torch.tensor(1)

    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].float().sum()
            res.append(correct_k.mul_(100.0 / batch_size))
    return res


def load_state_dict(net, state_dict):
    print('----------load_state_dict----------')
    # print(net.state_dict().keys())
    # print(state_dict.keys())
    try:
        net.load_state_dict(state_dict)
        print('try: loaded')
    except RuntimeError as e:
        if 'Missing key(s) in state_dict:' in str(e):
            net.load_state_dict({
                key.replace('module.', '', 1): value
                for key, value in state_dict.items()
            })
            print('except: loaded')

def save_checkpoint(ckpt, is_best, checkpoint_path, fname):
    ckpt_path = os.path.join(checkpoint_path, fname)
    print(f'=> Saving checkpoint to {ckpt_path}.pth...')
    torch.save(ckpt, f'{ckpt_path}.pth')
    if is_best:
        print(f'=> Saving checkpoint to {ckpt_path}_best.pth...')
        torch.save(ckpt, f'{ckpt_path}_best.pth')

def adjust_learning_rate(lr, epoch, epochs):
    if epoch == (epochs // 2):
        return lr / 10
    elif epoch == (epochs * 3 // 4):
        return lr / 100
    else:
        return lr

##########################################################
#             FOR DISTRIBUTED TRAINING                   #
#########################################################

def get_num_gpus():
    """Number of GPUs on this node."""
    return torch.cuda.device_count()


def get_local_rank():
    if 'MV2_COMM_WORLD_LOCAL_RANK' in os.environ:
        return int(os.environ['MV2_COMM_WORLD_LOCAL_RANK'])
    elif 'OMPI_COMM_WORLD_LOCAL_RANK' in os.environ:
        return int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
    elif 'SLURM_LOCALID' in os.environ:
        return int(os.environ['SLURM_LOCALID'])
    else:
        return 0

def get_local_size():
    if 'MV2_COMM_WORLD_LOCAL_SIZE' in os.environ:
        return int(os.environ['MV2_COMM_WORLD_LOCAL_SIZE'])
    elif 'OMPI_COMM_WORLD_LOCAL_SIZE' in os.environ:
        return int(os.environ['OMPI_COMM_WORLD_LOCAL_SIZE'])
    elif 'SLURM_NTASKS_PER_NODE' in os.environ:
        return int(os.environ['SLURM_NTASKS_PER_NODE'])
    else:
        return 1

def get_world_rank():
    if 'MV2_COMM_WORLD_RANK' in os.environ:
        return int(os.environ['MV2_COMM_WORLD_RANK'])
    elif 'OMPI_COMM_WORLD_RANK' in os.environ:
        return int(os.environ['OMPI_COMM_WORLD_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        return int(os.environ['SLURM_PROCID'])
    else:
        return 0

def get_world_size():
    if 'MV2_COMM_WORLD_SIZE' in os.environ:
        return int(os.environ['MV2_COMM_WORLD_SIZE'])
    elif 'OMPI_COMM_WORLD_SIZE' in os.environ:
        return int(os.environ['OMPI_COMM_WORLD_SIZE'])
    elif 'SLURM_NTASKS' in os.environ:
        return int(os.environ['SLURM_NTASKS'])
    else:
        return 1
    
def initialize_dist(init_file):
    """Initialize PyTorch distributed backend."""
    torch.cuda.init()
    torch.cuda.set_device(get_local_rank())
    init_file = os.path.abspath(init_file)
    torch.distributed.init_process_group(
        backend='nccl', init_method=f'file://{init_file}',
        rank=get_world_rank(), world_size=get_world_size())
    torch.distributed.barrier()
    # Ensure the init file is removed.
    if get_world_rank() == 0 and os.path.exists(init_file):
        os.unlink(init_file)

def get_cuda_device():
    return torch.device(f'cuda:{get_local_rank()}')

def allreduce_tensor(t):
    rt = t.clone().detach()
    torch.distributed.all_reduce(rt)
    rt /= get_world_size()
    return rt