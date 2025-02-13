import torch
import torch.nn as nn
import torch.distributed as dist


def multi_gpu_available() -> bool:
    return torch.cuda.is_available() and torch.cuda.device_count() > 1

def get_backend() -> str:
    if multi_gpu_available():
        return "nccl"
    else:
        return "gloo"

def sync_module_params(module: nn.Module):
    for param in module.parameters():
        dist.broadcast(param.data, src=0)
    
def gen_random_tensor_at_0(*size):
    if dist.get_rank() == 0:
        return torch.rand(*size)
    return None