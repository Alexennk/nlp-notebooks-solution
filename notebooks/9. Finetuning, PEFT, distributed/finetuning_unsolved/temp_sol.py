import os

import torch
import torch.distributed as dist
import torch.nn as nn

from utils import get_backend, sync_module_params, gen_random_tensor_at_0

# torchrun --nproc-per-node 2 temp_sol.py


if __name__ == "__main__":

    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend=get_backend(), rank=local_rank, world_size=world_size)

    """
    Иногда нам хочется послать тензор, но в принимающем процессе мы заранее не знаем, какого он размера.
    Давайте напишем функцию, которая вначале посылает двумерный тензор shape размерностей тензора, а потом
    уже его содержимое
    """

    dst = 1
    tensor = torch.Tensor([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
    ])
    

    if dist.get_rank() != dst:        
        dist.send(torch.tensor(tensor.shape, dtype=torch.int64), dst=dst)
        dist.send(tensor, dst=dst)
    else:
        shape = torch.zeros(2, dtype=torch.int64)
        dist.recv(tensor=shape, src=world_size - 1 - dst)
        tensor = torch.zeros(tuple(shape.tolist()))
        dist.recv(tensor=tensor, src=world_size - 1 - dst)
        print(f"Ранг текущего процесса: {dist.get_rank()}. Полученный тензор: {tensor}")
    
