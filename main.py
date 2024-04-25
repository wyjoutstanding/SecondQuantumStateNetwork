import os
import argparse
import logging
import numpy as np
import torch.multiprocessing as mp

from config import get_cfg_defaults  # local variable usage pattern, or:
# from config import cfg  # global singleton usage pattern
from src.util import prepare_dirs, set_seed, write_file, folder_name_generator
from src.helper import setup, cleanup



def run(cfg, local_rank, global_rank):
    mode = cfg.MISC.MODE
    best_score, time_elapsed, dic = 0, 0, None
    if mode in ['train']:
        from src.vmc.train import train
        best_score, time_elapsed, dic = train(cfg, local_rank, global_rank)
    return best_score, time_elapsed, dic


def run_trials(rank, cfg):
    local_rank = rank
    # set up configurations
    mode = cfg.MISC.MODE
    directory = cfg.MISC.DIR
    num_sites = cfg.DATA.NUM_SITES
    molecule_name = cfg.DATA.MOLECULE
    num_trials = cfg.MISC.NUM_TRIALS
    random_seed = cfg.SYSTEM.RANDOM_SEED
    result_logger_name = cfg.EVAL.RESULT_LOGGER_NAME
    node_idx = cfg.DDP.NODE_IDX
    local_world_size = cfg.DDP.LOCAL_WORLD_SIZE
    world_size = cfg.DDP.WORLD_SIZE
    global_rank = node_idx * local_world_size + local_rank
    master_addr = cfg.DDP.MASTER_ADDR
    master_port = cfg.DDP.MASTER_PORT
    use_same_local_seed = cfg.MISC.SAME_LOCAL_SEED
    logging.info(f"Running DDP on rank {global_rank}.")
    if world_size > 1:
        setup(global_rank, world_size, master_addr, master_port)
    if mode != 'debug' and global_rank == 0:
        prepare_dirs(cfg)
    avg_score = 0.0
    avg_time_elapsed = 0.0
    result_dic = {}
    write_file(result_logger_name, f"=============== {directory.split('/')[-1]} ===============", global_rank)
    for trial in range(num_trials):
        seed = random_seed + trial * 1000
        # set random seeds
        set_seed(seed + (0 if use_same_local_seed else global_rank))
        best_score, time_elapsed, dic = run(cfg, local_rank, global_rank)
        best_score = np.array(best_score)
        result_log = f"[{molecule_name}{num_sites}] Best Score {best_score}, Time elapsed {time_elapsed:.4f}"
        write_file(result_logger_name, f"Trial - {trial+1}", global_rank)
        write_file(result_logger_name, result_log, global_rank)
        avg_score += best_score / num_trials
        avg_time_elapsed += time_elapsed / num_trials
        if dic is not None:
            for key in dic:
                if key in result_dic:
                    result_dic[key] = np.concatenate((result_dic[key], np.expand_dims(dic[key], axis=0)), axis=0)
                else:
                    result_dic[key] = np.expand_dims(dic[key], axis=0)
    result_log = f"[{directory.split('/')[-1]}][{molecule_name}-{num_sites}] Avg Score {avg_score}, Time elapsed {avg_time_elapsed:.4f}, over {num_trials} trials"
    print()
    write_file(result_logger_name, result_log, global_rank)
    if mode != 'debug' and global_rank == 0:
        np.save(os.path.join(directory, 'result.npy'), result_dic)
    if world_size > 1:
        cleanup()


def main(cfg):
    world_size = cfg.DDP.WORLD_SIZE
    local_world_size = cfg.DDP.LOCAL_WORLD_SIZE
    if world_size > 1:
        mp.spawn(run_trials, args=(cfg,), nprocs=local_world_size, join=True)
    else:
        run_trials(0, cfg)
    logging.info('--------------- Finished ---------------')
    

if __name__ == '__main__':
    # command line arguments
    parser = argparse.ArgumentParser(description="Command-Line Options")
    parser.add_argument(
        "--config_file",
        default="",
        metavar="FILE",
        help="Path to the yaml config file",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    # configurations
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # set up directories (cfg.MISC.DIR)
    if cfg.MISC.DIR == '':
        cfg.MISC.DIR = './logger/{}'.format(folder_name_generator(cfg, args.opts))
    os.makedirs(cfg.MISC.DIR, exist_ok=True)
    # freeze the configurations
    cfg.freeze()
    # set logger
    logging.root.handlers = []
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(cfg.MISC.DIR, 'logging.log')),
            logging.StreamHandler()
        ]
    )
    # run program
    main(cfg)
    