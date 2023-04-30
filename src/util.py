
import os
import random
import logging
import numpy as np
import torch
from datetime import datetime


def set_seed(seed):
    # the followings are for reproducibility on GPU, see https://pytorch.org/docs/master/notes/randomness.html
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_time():
    return datetime.now().strftime("%m%d_%H%M%S")

def folder_name_generator(cfg, opts):
    name_str = []
    name_str.append('{}'.format(cfg.MISC.MODE))
    for i,arg in enumerate(opts):
        if i % 2 == 1:
            if opts[i-1] == 'DATA.LOAD_PATH':
                continue
            name_str.append('{}'.format(arg))
    return '-'.join(name_str)

def prepare_dirs(cfg):
    if not os.path.exists('./datasets'):
        os.makedirs('./datasets')
    if not os.path.exists('./results'):
        os.makedirs('./results')
    if not os.path.exists('./logger'):
        os.makedirs('./logger')
    if not os.path.exists(cfg.MISC.DIR):
        os.makedirs(cfg.MISC.DIR)
    # set logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(cfg.MISC.DIR, 'debug.log')),
            logging.StreamHandler()
        ]
    )

def write_file(file_name, content, local_rank=0):
    # only write to disk on the master thread
    if local_rank == 0:
        f=open(file_name, "a+")
        f.write(content)
        f.write("\n")
        f.close()

