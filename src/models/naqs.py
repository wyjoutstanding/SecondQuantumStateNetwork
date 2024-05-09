import math
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from .base import Base

def multinomial_arr(count, p):
    # count: [num_uniqs], p: [num_uniqs, probs]
    # out: [num_uniqs, # samples]
    count = np.copy(count)
    out = np.zeros(p.shape, dtype=int)
    ps = p.cumsum(axis=-1)
    with np.errstate(divide='ignore', invalid='ignore'):
        condp = p / ps
    condp[np.isnan(condp)] = 0.0
    for i in range(p.shape[-1] - 1, 0, -1):
        binsample = np.random.binomial(count, condp[:, i])
        out[:, i] = binsample
        count -= binsample
    out[:, 0] = count
    return out


class SoftmaxLogProbAmps(nn.Module):
    def __init__(self, ablation_idx):
        super(SoftmaxLogProbAmps, self).__init__()
        self.ablation_idx = ablation_idx

    def mask_input(self, x, mask, val):
        m = mask.clone().to(x.device)
        x_ = x.masked_fill((1 - m).bool(), val)
        return x_

    def forward(self, x, mask=None, dim=1):
        x_ = self.mask_input(2*x, mask, float('-inf'))
        # x_ = self.mask_input(x, mask, float('-inf')) this is a more intuitive approach
        return 0.5 * F.log_softmax(x_, dim=dim)


class OrbitalBlock(nn.Module):
    def __init__(self, num_in=2, n_hid=[]):
        super(OrbitalBlock, self).__init__()
        self.num_in = num_in
        self.n_hid = n_hid
        self.num_out = 4
        layer_dims = [num_in] + n_hid + [self.num_out]
        self.layers = []
        for i, (n_in, n_out) in enumerate(zip(layer_dims[:-1], layer_dims[1:])):
            l = [nn.Linear(n_in, n_out, bias=True)]
            if i < len(layer_dims) - 2:
                l.append(nn.ReLU())
            l = nn.Sequential(*l)
            self.layers.append(l)
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.layers(x)


class NADE(Base):
    def __init__(self, num_sites, num_spin_up, num_spin_down, hidden_size, hidden_depth, ablation_idx):
        super(NADE, self).__init__()
        self.num_sites = num_sites
        self.amp_layers, self.phase_layers = [], []
        for n in range(self.num_sites // 2):
            inp_dim = max(1, 2*n) # One dimensional input for the very first block
            amp_i = OrbitalBlock(num_in=inp_dim, n_hid=[hidden_size]*hidden_depth)
            self.amp_layers.append(amp_i)
            if n == self.num_sites // 2 - 1:
                phase_i = OrbitalBlock(num_in=inp_dim, n_hid=[hidden_size*8]*(hidden_depth+1))
                self.phase_layers.append(phase_i)
        self.amp_layers = nn.ModuleList(self.amp_layers)
        self.phase_layers = nn.ModuleList(self.phase_layers)
        self.amp_act = SoftmaxLogProbAmps(ablation_idx=ablation_idx)
        self.sampling = False
        self.num_spin_up, self.num_spin_down = num_spin_up, num_spin_down

        self.qubit2model_permutation = torch.stack([torch.arange(self.num_sites-2,-1,-2), torch.arange(self.num_sites-1,-1,-2)],1).reshape(-1)
        self.model2qubit_permutation = np.argsort(self.qubit2model_permutation)
        self.state2model_permutation_shell = torch.arange(self.num_sites//2-1, -1, -1)
        self.model2state_permutation_shell = np.argsort(self.state2model_permutation_shell)

        self.ablation_idx = ablation_idx


    def process_inputs(self, x, i):
        bs = x.shape[0]
        if i == 0:
            x_inp = torch.zeros(bs, 1)
            x_alpha, x_beta = torch.zeros(bs, 1), torch.zeros(bs, 1)
        else:
            x_alpha, x_beta = x[:, :2*i:2].clone(), x[:, 1:2*i:2].clone()
            # x_inp = x[:, :2*i].clone() # this is a more intuitive approach
            x_inp = torch.cat([x_alpha, x_beta], -1)
        return x_inp.to(x.device), x_alpha.to(x.device), x_beta.to(x.device)

    @torch.no_grad()
    def get_constraint_mask(self, x_alpha, x_beta, i):
        threshold = np.min([self.num_spin_up, self.num_spin_down, self.num_sites-self.num_spin_up, self.num_sites-self.num_spin_down])
        if i >= max(threshold, 1):
            # Recall amp ordering of [|0,0>, |1,0>, |0,1>, |1,1>]
            is_alpha_down_idxs = torch.LongTensor([0, 2])
            is_alpha_up_idxs = torch.LongTensor([1, 3])
            is_beta_down_idxs = torch.LongTensor([0, 1])
            is_beta_up_idxs = torch.LongTensor([2, 3])
            _num_spin_up_up, _num_spin_up_down, _num_spin_down_up, _num_spin_down_down = self.num_spin_up, math.ceil(self.num_sites/2)-self.num_spin_up, self.num_spin_down, math.floor(self.num_sites/2)-self.num_spin_down
            num_spin_up_up, num_spin_down_up = (x_alpha > 0).sum(1), (x_beta > 0).sum(1)
            num_spin_up_down, num_spin_down_down = x_alpha.shape[-1]-num_spin_up_up, x_beta.shape[-1]-num_spin_down_up
            set_alpha_down_idxs = torch.where(num_spin_up_up >= _num_spin_up_up)[0]
            set_alpha_up_idxs = torch.where(num_spin_up_down >= _num_spin_up_down)[0]
            set_beta_down_idxs = torch.where(num_spin_down_up >= _num_spin_down_up)[0]
            set_beta_up_idxs = torch.where(num_spin_down_down >= _num_spin_down_down)[0]
            mask = torch.ones(len(x_alpha), 4)
            for set_idx, is_idx in zip(
                    [set_alpha_down_idxs, set_alpha_up_idxs, set_beta_down_idxs, set_beta_up_idxs],
                    [is_alpha_up_idxs, is_alpha_down_idxs, is_beta_up_idxs, is_beta_down_idxs]):
                mask[set_idx.repeat_interleave(len(is_idx)), is_idx.repeat(len(set_idx))] = 0
        else:
            mask = torch.ones(len(x_alpha), 4)
        return mask

    def state2shell(self, states):
        # convert [|0,0>, |1,0>, |0,1>, |1,1>] to [0, 1, 2, 3]
        bs = states.shape[0]
        shell_size = 2
        shell = (states.view(bs, -1, shell_size).clamp_min(0) * torch.Tensor([1.0, 2.0]).to(states.device)).sum(-1)
        return shell.type(torch.int64)

    def predict(self, x):
        num_spins = x.shape[-1]
        outputs = []
        for i in range(num_spins // 2):
            x_inp, x_alpha, x_beta = self.process_inputs(x, i)
            amp_i = self.amp_layers[i](x_inp)
            if i == (self.num_sites // 2 - 1):
                phase_i = self.phase_layers[0](x_inp)
            else:
                phase_i = torch.zeros_like(amp_i).to(x.device)
            amp_mask = self.get_constraint_mask(x_alpha, x_beta, i)
            amp_i = self.amp_act(amp_i, amp_mask)
            amp_i[amp_mask.sum(-1)==0] = float('-inf')
            out_i = torch.stack([amp_i, phase_i], -1)
            outputs.append(out_i)
        return torch.stack(outputs, 1)

    def sample(self, bs):
        self.sampling = True
        device = self.amp_layers[0].layers[0][0].weight.device
        states = torch.zeros(1, 2, requires_grad=False)
        probs = torch.FloatTensor([1]).to(device)
        counts = torch.LongTensor([int(bs)])
        blockidx2spin = torch.FloatTensor([[-1, -1], [1, -1], [-1, 1], [1, 1]])
        for i in range(self.num_sites // 2):
            x_inp, x_alpha, x_beta = self.process_inputs(states.to(device), i)
            amp_i = self.amp_layers[i](x_inp)
            amp_mask = self.get_constraint_mask(x_alpha, x_beta, i)
            amp_i = self.amp_act(amp_i, amp_mask)
            amp_i[amp_mask.sum(-1)==0] = float('-inf')
            with torch.no_grad():
                probs_i = amp_i.detach().exp().pow(2)
                next_probs = probs.unsqueeze(1) * probs_i.to(probs)
                probs_i_np = probs_i.cpu().numpy().astype('float64')
                probs_i_np /= np.sum(probs_i_np, -1, keepdims=True)
                new_sample_counts = torch.LongTensor(multinomial_arr(counts, probs_i_np))
                new_sample_counts *= amp_mask.to(new_sample_counts)
                new_sample_mask = (new_sample_counts > 0)
                num_new_samples_per_state = new_sample_mask.sum(1)
                new_sample_next_idxs = torch.where(new_sample_counts > 0)[1]
                if i == 0:
                    states = blockidx2spin[new_sample_next_idxs]
                else:
                    states = torch.cat([states.repeat_interleave(num_new_samples_per_state, 0), blockidx2spin[new_sample_next_idxs]], 1)
                counts = new_sample_counts[new_sample_mask]
                probs = next_probs[new_sample_mask]
        states = states[:, self.model2qubit_permutation]
        return states.to(device), counts, None, None

    def forward(self, x):
        self.sampling = False
        bs = x.shape[0]
        # log_psi_cond = self.predict(x.to(self.device)) this is a more intuitive approach
        log_psi_cond = self.predict(x[..., self.qubit2model_permutation].to(x.device))[:, self.model2state_permutation_shell, ...] # [bs, num_shell, num_prob, 2]
        log_psi_cond = log_psi_cond.gather(-2, self.state2shell(x).view(bs, -1, 1, 1).repeat(1, 1, 1, 2))
        log_psi = log_psi_cond.sum(axis=1).squeeze()
        return log_psi

    def set_mask_device(self):
        pass

