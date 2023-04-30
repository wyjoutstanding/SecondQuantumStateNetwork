import os
import numpy as np
from functools import wraps
import torch.distributed as dist


def suspendlogging(func):
    @wraps(func)
    def inner(*args, **kwargs):
        previousloglevel = log.getEffectiveLevel()
        try:
            return func(*args, **kwargs)
        finally:
            log.setLevel(previousloglevel)
    return inner

def setup(global_rank, world_size, master_addr, master_port):
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = str(master_port)
    # initialize the process group
    dist.init_process_group("nccl", rank=global_rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def sort_strlst(lst):
    lst_copy = []
    for i in lst:
        lst_copy.append(int(i))
    idx = np.argsort(np.array(lst_copy))
    return list(np.array(lst)[idx])

def idx_to_coord(idx, side_len):
    row = int(idx // side_len)
    col = int(idx % side_len)
    return row, col

def coord_to_idx(coord, side_len):
    row, col = coord
    idx = int(row * side_len + col)
    return idx
