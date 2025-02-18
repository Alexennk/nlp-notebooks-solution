import os

import torch
import torch.distributed as dist

from utils import get_backend

# print_rank_0 - 5 баллов
def print_rank_0(message):
    """
    Зачастую нам хочется что-то вывести, однако весь код выполняется на всех процессах,
    поэтому вывод сообщений дублируется. Допишите функцию print_rank_0, которая
    выводила бы сообщение только на нулевом (главном) процессе.
    
    После этого нужно синхронизировать процессы, т.к. в данном случае мы не хотим,
    чтобы другие процессы уходили в другие функции, а дождались, пока главный процесс
    допечатает сообщение.

    Нужно использовать dist.get_rank и dist.barrier
    """

    if dist.get_rank() == 0:
        print(message)
    dist.barrier() # процессы, достигшие барьера, ожидают всех остальных
    # raise NotImplemented()


# blocking_send_to_last - 5 баллов
def blocking_send_to_last():
    """
    Ваша задача послать с каждого процесса, кроме последнего, свой ранг последнему процессу.
    Последний процесс должен получить ранги всех остальных процессов и сложить их.

    Для пересылки нужно использовать блокирующий dist.send.
    Для получения нужно использовать блокирующий dist.recv. Обратите внимание, что в recv
    аргумент src не обязателен!
    
    Документация https://pytorch.org/docs/stable/distributed.html#point-to-point-communication
    
    """

    send_value = torch.Tensor([dist.get_rank()]).long()
    sum_values = torch.zeros(1).long()

    if dist.get_rank() != world_size - 1:
        dist.send(tensor=send_value, dst=world_size - 1, group=None, tag=0, group_dst=None)
    elif dist.get_rank() == world_size - 1:
        recv_value = torch.zeros(1).long()
        for i in range(world_size - 1):
            sum_values += dist.recv(tensor=recv_value, src=i, group=None, tag=0, group_src=None)

    print(f"Ранг текущего процесса: {dist.get_rank()}. Вычисленная сумма: {sum_values}")
    dist.barrier()
    print_rank_0("Успешно послали свои ранги последнему процессу\n")


# cyclic send-recv - 5 баллов
def cyclic_send_recv():
    """
    В этой задаче вам необходимо послать значение send_tensor следующему процессу (ранг + 1) от текущего
    процесса и соответственно принять результат посылки от предыдущего процесса (ранг - 1).
    Т.е. 0й посылает тензор 1му, 1й процесс посылает данные 2му, 2й посылает 3му, а 3й посылает 0му.
    Аналогично с получениями: 0й получает от 3го, 1й от 0го, 2й от 1го и 3й от 2го.

    Для посылки и принятия результатов используйте асинхронные функции dist.isend, dist.irecv.
    Эти функции возвращают объект Work, у которого есть метод .wait() - он позволяет дождаться конца
    коммуникации, которая его породила.

    Документация https://pytorch.org/docs/stable/distributed.html#torch.distributed.isend
    
    """
    values_to_send = [10, 20, 30, 40]
    values_to_recv = [40, 10, 20, 30]
    send_tensor = torch.Tensor([values_to_send[dist.get_rank()]])
    recv_tensor = torch.zeros_like(send_tensor)
    rank = dist.get_rank()

    send_req = dist.isend(tensor=send_tensor, dst=(rank + 1) % world_size)
    recv_req = dist.irecv(tensor=recv_tensor, src=(rank - 1) % world_size)
    send_req.wait()
    recv_req.wait()
    
    print(f"Ранг текущего процесса: {dist.get_rank()}. Полученный тензор: {recv_tensor}. Ожидаемое значение: {values_to_recv[dist.get_rank()]}.\n")
    dist.barrier()
    print_rank_0("Процессы успешно получили тензоры соседних процессов!")


# group_comms - 5 баллов
def group_comms():
    """
    На каждом ранге гененрируется случайный тензор.
    Ваша задача:
    1. С помощью операции all_reduce найти минимальное значение среди всех local_tensor
    2. Собрать все local_tensor на всех процессах с помощью all_gather и найти минимальное значение
    """
    # 1 часть с all_reduce
    local_tensor = torch.rand(1)
    print(f"Ранг текущего процесса: {dist.get_rank()}. Случайный тензор: {local_tensor}\n")
    dist.all_reduce(local_tensor, op=dist.ReduceOp.MIN)
    print_rank_0(f"Ранг текущего процесса: {dist.get_rank()}. Минимальный тензор: {local_tensor}\n")

    # 2 часть с all_gather
    local_tensor = torch.rand(1)
    tensor_list = [torch.zeros(1) for _ in range(world_size)]
    dist.all_gather(tensor_list, local_tensor)
    min_tensor = torch.min(torch.stack(tensor_list))
    print_rank_0(f"Ранг текущего процесса: {dist.get_rank()}. Список тензоров: {tensor_list}. Минимальный тензор: {min_tensor}\n")

    print_rank_0("Успешно провели групповые коммуникации!")
    


if __name__ == "__main__":
    # данное задание предполагает запуск в 4 процесса
    # torchrun --nproc-per-node 4 2_communications.py
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend=get_backend(), rank=local_rank, world_size=world_size)

    print_rank_0("Это сообщение должно быть выведено всего один раз")
    blocking_send_to_last()
    cyclic_send_recv()
    group_comms()

    # в файле output.txt вывод программы