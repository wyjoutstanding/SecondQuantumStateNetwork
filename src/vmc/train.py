import os
from pickle import NONE
from shelve import BsdDbShelf
import pickle
import time
import logging
from tqdm import trange
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel
from tensorboardX import SummaryWriter

from backpack import extend

from src.optimizers.sr import SR
from src.data_loader import load_data
from src.scheduler import get_scheduler
from src.optimizer import get_optimizer
from src.models.util import get_model, load_model, save_model, get_model_gradient
from src.complex import real, imag, scalar_mult

from .evaluate import test
from .objective.util import get_hamiltonian
from .sampler.flow_sampler import FLOWSampler
from src.optimizers.util import apply_grad, vec_to_grad


MCMC_MODELS = ['rbm', 'rbm_c']

def train_one_batch(model, sampler, hamiltonian, optimizer, scheduler, sr, batch_size, num_samples, ablation_idx, inner_iter, global_rank, world_size, temp_path, nmlzr, epoch):
    model.train()
    losses = {}
    device = list(model.parameters())[0].device
    # collect samples
    samples = sampler(batch_size*world_size, num_samples)
    if samples[0].shape[0] < batch_size*world_size:
        batch_size = np.ceil(samples[0].shape[0] / world_size).astype(np.int64)
    if isinstance(samples, list):
        samples, count = samples
        weight = count / count.sum()
    else:
        bs = samples.shape[0]
        samples = torch.tensor(samples).float().to(device)
        weight = torch.ones([bs]).to(samples.device) / bs
    # get the corresponding batch for each GPU device
    partition = world_size - global_rank - 1
    samples = samples[partition*batch_size:(partition+1)*batch_size]
    weight = weight[partition*batch_size:(partition+1)*batch_size]
    total_local_energies = []
    total_losses = []
    inner_iter = min(inner_iter, samples.shape[0])
    # save the gradient of each small batch_size to disk and then sum them up for updates
    for i in range(inner_iter):
        sbs = np.ceil(samples.shape[0] / inner_iter).astype(np.int64)
        sbs_samples = samples[i*sbs:(i+1)*sbs]
        sbs_weight = weight[i*sbs:(i+1)*sbs]
        if sr is not None:
            model = extend(model)
        # train
        local_energies, log_psi = hamiltonian.compute_local_energy(sbs_samples, model)
        log_psi_conj = torch.stack((log_psi[:, 0], -log_psi[:, 1]), dim=-1)
        wt = sbs_weight.unsqueeze(-1)
        loss = 2 * (real(scalar_mult(log_psi_conj, (local_energies - nmlzr).detach()) * wt)).sum()
        # if epoch > 2000:
        #     import pdb;pdb.set_trace()
        # if sbs_weight.max() > 0.9993:
        #     import pdb;pdb.set_trace()
        # print(sbs_weight.max(), sbs_weight.min())
        loss.backward()
        total_local_energies.append(local_energies)
        total_losses.append(loss.item())
    
    optimizer.step()
    scheduler.step()
    # print("Learning rate:", optimizer.param_groups[0]['lr'])
    optimizer.zero_grad()
    losses['loss'] = np.sum(total_losses)
    local_energies = torch.cat(total_local_energies, axis=0)
    mean_energy = (local_energies * weight.unsqueeze(-1)).sum(0)
    if global_rank == 0:
        score = real(mean_energy.detach())
        scores = real(local_energies.detach())
        std = ((scores - score)**2 * wt).sum().sqrt()
        max_score = scores.min()
        losses['score'] = score.item()
        losses['std'] = std.item()
        losses['max_score'] = max_score.item()
        losses['num_uniq'] = samples.shape[0]
    return losses



def train(cfg, local_rank, global_rank):
    # set hyper-parameters
    # settings
    data_path = cfg.DATA.LOAD_PATH
    logger_dir = cfg.MISC.DIR
    eval_itvl = cfg.EVAL.EVALUATION_INTERVAL
    device = torch.device('cuda:{}'.format(local_rank) if (cfg.SYSTEM.NUM_GPUS > 0) else 'cpu')
    world_size = cfg.DDP.WORLD_SIZE
    # train
    lr = cfg.TRAIN.LEARNING_RATE
    num_epochs = cfg.TRAIN.NUM_EPOCHS
    bs = cfg.TRAIN.BATCH_SIZE
    opt_name = cfg.TRAIN.OPTIMIZER_NAME
    sche_name = cfg.TRAIN.SCHEDULER_NAME
    apply_sr = cfg.TRAIN.APPLY_SR
    sr = SR(damping_factor=0.001) if apply_sr else None
    inner_iter = cfg.TRAIN.INNER_ITER
    # AUTO
    num_samples = cfg.DATA.NUM_SAMPLES
    # model
    model_name = cfg.MODEL.MODEL_NAME
    model_load_path = cfg.EVAL.MODEL_LOAD_PATH
    depth = cfg.MODEL.HIDDEN_DEPTH
    width = cfg.MODEL.HIDDEN_WIDTH
    # load data
    data = load_data(cfg)
    num_sites = data['molecule'].n_qubits
    assert data['molecule'].n_qubits == 2*data['molecule'].n_orbitals
    assert (data['molecule'].n_qubits-data['molecule'].n_electrons) % 2 == 0
    num_spin = int((data['molecule'].n_qubits-data['molecule'].n_electrons)/2)
    # num_spin_up = cfg.MODEL.NUM_SPIN_UP
    # num_spin_down = cfg.MODEL.NUM_SPIN_DOWN
    num_spin_up = num_spin
    num_spin_down = num_spin
    logging.info('Num of up/down spins {}/{}'.format(num_spin_up, num_spin_up))
    # debug
    ablation_idx = cfg.DEBUG.ABLATION_IDX
    # load model
    model = get_model(model_name, device,
                      print_model_info=True,
                      num_sites=num_sites,
                      hidden_size=width,
                      hidden_depth=depth,
                      num_spin_up=num_spin_up,
                      num_spin_down=num_spin_down,
                      ablation_idx=ablation_idx,
                      )
    if model_load_path:
        model = load_model(model, model_load_path)
    # set up training
    hamiltonian = get_hamiltonian(**data)
    hamiltonian.set_device(device)
    optimizer = get_optimizer(opt_name, model, lr)
    scheduler = get_scheduler(sche_name, optimizer, lr, num_epochs)
    model.set_mask_device()
    sampler = FLOWSampler(model, bs, num_sites)
    if world_size > 1:
        print("The local rank here is {}".format(local_rank))
        model = DistributedDataParallel(model, device_ids=[local_rank])
    # tensorboard
    if global_rank == 0:
        tensorboard = SummaryWriter(log_dir=logger_dir)
        tensorboard.add_text(tag='argument', text_string=str(cfg.__dict__))
    # train
    temp_path = os.path.join('./temp', os.path.basename(data_path))
    if not os.path.exists(temp_path):
        os.makedirs(temp_path)
    best_mean, best_max = 0.0, 0.0
    time_elapsed = 0.0
    dic = {'mean': [], 'max': [], 'std': [], 'num_uniq': []}
    if global_rank == 0:
        progress_bar = trange(num_epochs, desc='Progress Bar', leave=True)
    for epoch in range(1, num_epochs + 1):
        # evaluation
        test_bs = int(bs*world_size)
        nmlzr, mean, std, max, avg_accept, num_uniq = test(model, sampler, hamiltonian, test_bs, num_samples, inner_iter, world_size)
        # train
        start_time = time.time()
        losses = train_one_batch(model, sampler, hamiltonian, optimizer, scheduler, sr, bs, num_samples, ablation_idx, inner_iter, global_rank, world_size, temp_path, nmlzr, epoch)
        end_time = time.time()
        time_elapsed += end_time - start_time
        if global_rank == 0:
            for key in losses:
                tensorboard.add_scalar('train/{}'.format(key), losses[key].real, epoch)
        # log
        message = '[Epoch {}]'.format(epoch)
        for key in losses:
            if key in ['loss']:
                message += ' {}: {:.6f}'.format(key, abs(losses[key]))
        dic['mean'].append(mean)
        dic['max'].append(max)
        dic['std'].append(std)
        dic['num_uniq'].append(num_uniq)
        if global_rank == 0:
            tensorboard.add_scalar('test/mean', mean, epoch)
            tensorboard.add_scalar('test/max', max, epoch)
            tensorboard.add_scalar('test/std', std, epoch)
            tensorboard.add_scalar('test/num_uniq', num_uniq, epoch)
        message += ', [Test] mean/max/std/ Score: {:.6f}/{:.6f}/{:.6f}, {} Uniqs'.format(mean, max, std, num_uniq)
        if model_name in MCMC_MODELS:
            message += ', AR: {:.2f}'.format(avg_accept)
        if mean < best_mean:
            best_mean = mean
            best_max = max
        if global_rank == 0:
            progress_bar.set_description(message)
            progress_bar.refresh() # to show immediately the update
            progress_bar.update(1)
        if epoch % int(num_epochs/5) == 0:
            logging.info(message)
    if global_rank == 0:
        tensorboard.close()
    best_mean = np.median(dic['mean'][-len(dic['mean'])//20:])
    return [best_mean, best_max], time_elapsed, dic