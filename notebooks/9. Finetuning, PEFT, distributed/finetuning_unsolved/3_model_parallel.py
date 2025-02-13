import os

import torch
import torch.distributed as dist
import torch.nn as nn

from utils import get_backend, sync_module_params, gen_random_tensor_at_0


def send_2d_tensor(tensor, dst):
    """
    Иногда нам хочется послать тензор, но в принимающем процессе мы заранее не знаем, какого он размера.
    Давайте напишем функцию, которая вначале посылает двумерный тензор shape размерностей тензора, а потом
    уже его содержимое
    """
    raise NotImplemented()


def recv_2d_tensor(src):
    """
    Эта функция должна
    1. Принимать двумерный тензор размерностей
    2. Создавать тензор нужных размерностей для приема данных
    3. Принимать данные в этот тензор и возвращать его
    """
    raise NotImplemented()


class PipeliningLinearLayer(nn.Module):

    def __init__(self):
        super().__init__()
        self.ln1 = nn.Linear(16, 32)
        self.ln2 = nn.Linear(32, 64)
    
    def forward(self, x):
        """
        Здесь нужно дописать логику, описанную в test_pipelining
        
        """
        raise NotImplemented()
        

    def forward_full_rank_0(self, x):
        if dist.get_rank() == 0:
            return self.ln2(self.ln1(x))
        return None


# 5 баллов
def test_pipelining():
    """
    Ваша задача дописать PipeliningLinearLayer. Обратите внимание, что входной тензор есть только у 0го процесса!
    Требуется:
    1. На 0м процессе применить первый линейный слой к входному тензору
    2. Послать результат на 1й процесс
    3. На 1м процессе принять результат и вернуть его. На 0м процессе вернуть None

    Т.к. мы пересылаем тензоры неизвестных размеров, нужно дописать и использовать функции 
    send_2d_tensor и recv_2d_tensor.
    """
    pp_layer = PipeliningLinearLayer()
    sync_module_params(pp_layer)
    pp_input = gen_random_tensor_at_0(7, 16)
    pp_output = pp_layer(pp_input)


    if dist.get_rank() == 1:
        send_2d_tensor(pp_output, 0)
    else:
        pp_output = recv_2d_tensor(1)
        assert torch.allclose(pp_output, pp_layer.forward_full_rank_0(pp_input))
        print("Успешно отработал пайплайнинг")
    dist.barrier()



# 5 баллов
def test_tensor_parallel():
    """
    Здесь ваша задача реализовать tensor parallel для линейного слоя,
    т.е. реализовать операцию X @ A
    Для этого:
    1. Разбейте матрицу A по процессам по последней размерности
    2. Сделайте матричные умножения на X
    3. Сконкатенируйте результаты обратно
    """
    A = torch.rand(16, 32)
    dist.broadcast(A, 0)
    X = torch.rand(7, 16)
    dist.broadcast(X, 0)

    if dist.get_rank() == 0:
        raise NotImplemented()
    else:
        raise NotImplemented()
    
    Y_REF = X @ A

    assert torch.allclose(Y_REF, Y)
    print("Успешно отработал tensor parallel")
    


if __name__ == "__main__":
    # данное задание предполагает запуск в 2 процесса
    # torchrun --nproc-per-node 2 3_model_parallel.py
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend=get_backend(), rank=local_rank, world_size=world_size)
    test_pipelining()
    test_tensor_parallel()
