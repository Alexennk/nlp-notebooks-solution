import os

import torch
import torch.distributed as dist

from utils import get_backend


if __name__ == "__main__":
    """
    Для запуска данного скрипта нужно написать
    torchrun --nproc-per-node 2 1_hello_distributed.py
    Тогда запускаются ряд процессов, которые могут между собой коммуницировать.
    """
    # локальный ранг процесса - его порядковый номер на данной машине. Еще есть просто ранг
    # это глобальный ранг между всеми машинами, но т.к. у нас локальный запуск, то глобальный ранг
    # соответствует локальному
    local_rank = int(os.environ["LOCAL_RANK"])
    # world_size - размер мира, т.е. число запущенных процессов
    world_size = int(os.environ["WORLD_SIZE"])

    # подробнее про переменные окружения можно прочитать здесь https://pytorch.org/docs/stable/elastic/run.html#environment-variables


    # инициализация группы процессов. Все процессы до данной строчки запущены независимо, но после нее
    # устанавливается метод коммуникации между ними и теперь процессы могут обмениваться сообща
    # backend - или nccl, если есть > 1 gpu или gloo для cpu
    dist.init_process_group(backend=get_backend(), rank=local_rank, world_size=world_size)

    # dist == torch.distributed - наш модуль для всех взаимодействий между процессами
    # Порядок работы и вывода недетерменирован!
    print(f"Hello from process {dist.get_rank()} out of {dist.get_world_size()}\n")
    
