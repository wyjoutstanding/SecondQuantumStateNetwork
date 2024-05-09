import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Base
from .util import get_input_degrees, multinomial_arr


class MADE(Base):
    def __init__(self, num_sites, hidden_size, hidden_depth, num_spin_up, num_spin_down, **kwargs):
        super(MADE, self).__init__()
        self.num_sites = num_sites
        self.num_spin_up = num_spin_up
        self.num_spin_down = num_spin_down
        self.sampler_type = 'FLOW'
        # construct model
        self.net = []
        self.num_in, self.num_out = num_sites, num_sites*2
        self.hidden_sizes = [hidden_size] * hidden_depth
        layer_sizes = [self.num_in] + self.hidden_sizes + [self.num_out]
        for h0,h1 in zip(layer_sizes, layer_sizes[1:]):
            self.net.extend([
                    nn.Linear(in_features=h0, out_features=h1, bias=True),
                    nn.ReLU(),
                ])
        self.net.pop() # pop the last ReLU for the output layer
        self.net = nn.Sequential(*self.net)
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.net_phase = [nn.Linear(in_features=num_sites-2, out_features=hidden_size, bias=True)]
        self.net_phase += [nn.ReLU(), nn.Linear(in_features=hidden_size, out_features=4, bias=True)]
        self.net_phase = nn.Sequential(*self.net_phase)
        # create masks
        self.get_masks() # builds the initial self.m connectivity
        self.set_masks()
        # get_input_degrees(self.net, self.num_in, self.num_out)
        self.sampling = False

    def get_masks(self):
        self.m = {}
        self.seed = 0 # for cycling through num_masks orderings
        L = len(self.hidden_sizes)
        # fetch the next seed and construct a random stream
        rng = np.random.RandomState(self.seed)
        # sample the reverse order of the inputs and the connectivity of all neurons
        self.input_order = np.stack([np.arange(self.num_sites-2,-1,-2), np.arange(self.num_sites-1,-1,-2)],1).reshape(-1) # [4,5,2,3,0,1]
        self.shell_order = torch.arange(self.num_sites//2-1, -1, -1) # [2,1,0]
        self.m[-1] = self.input_order // 2 # [2,2,1,1,0,0]
        for l in range(L):
            self.m[l] = rng.randint(self.m[l-1].min(), self.num_sites//2-1, size=self.hidden_sizes[l])
        # construct the mask matrices
        masks = [self.m[l-1][:,None] <= self.m[l][None,:] for l in range(L)]
        masks.append(self.m[L-1][:,None] < np.repeat(self.m[-1], 2)[None,:])
        for i,_ in enumerate(masks):
            masks[i] = torch.tensor(np.array(masks[i]))
        self.masks = masks

    def set_masks(self):
        self.device = self.net[0].weight.device
        layers = [layer for layer in self.net.modules() if isinstance(layer, nn.Linear)]
        assert len(layers) == len(self.masks)
        for layer,mask in zip(layers, self.masks):
            layer.weight.data = layer.weight.data.mul_(mask.to(self.device).T)

    def state2shell(self, states):
        # convert [|0,0>, |1,0>, |0,1>, |1,1>] to [0, 1, 2, 3]
        bs = states.shape[0]
        shell_size = 2
        shell = (states.view(bs, -1, shell_size).clamp_min(0) * torch.Tensor([1.0, 2.0]).to(states.device)).sum(-1)
        return shell.type(torch.int64)

    def forward(self, x):
        # x: [bs, num_sites]
        # convert [|0,0>, |1,0>, |0,1>, |1,1>] to [0, 1, 2, 3]
        self.set_masks()
        logits_cls = self.net(x).reshape(-1, self.num_sites//2, 4) # bs, num_sites/2, 4
        if (self.num_spin_up + self.num_spin_down) >= 0:
            log_psi_cond = self.apply_constraint(x, logits_cls)
        # print(f"1 log_psi_cond: {log_psi_cond.shape} logits_cls: {logits_cls.shape}")
        log_psi_cond = 0.5 * self.log_softmax(logits_cls)
        # print(f"2 log_psi_cond: {log_psi_cond.shape} {log_psi_cond}")
        # print(f"log_psi_cond: {log_psi_cond}")
        if self.sampling:
            prob_cond = (2 * log_psi_cond).exp()
            return prob_cond
        else:
            idx = self.state2shell(x)
            log_psi_real = log_psi_cond.gather(-1, idx.unsqueeze(-1)).sum(-1).sum(-1)
            log_psi_imag = self.net_phase(x[:, :-2]).gather(-1, idx[:, -1].unsqueeze(-1)).squeeze()
            log_psi = torch.stack((log_psi_real, log_psi_imag), dim=-1)
            return log_psi

    def apply_constraint(self, inp, log_psi_cond):
        # convert [|-1,-1>, |1,-1>, |-1,1>, |1,1>] to [0, 1, 2, 3]
        device = inp.device
        N = inp.shape[-1] // 2
        inp_up = inp[:, self.input_order][:, ::2].clone()
        inp_down = inp[:, self.input_order][:, 1::2].clone()
        inp_cumsum_up = torch.cat((torch.zeros((inp_up.shape[0],1)).to(device), ((inp_up + 1)/2).cumsum(-1)[:, :-1]), axis=-1)
        inp_cumsum_down = torch.cat((torch.zeros((inp_down.shape[0],1)).to(device), ((inp_down + 1)/2).cumsum(-1)[:, :-1]), axis=-1)
        upper_bound_up = self.num_spin_up
        lower_bound_up = (self.num_spin_up - (N - torch.arange(1, N+1)))
        condition1_up = (inp_cumsum_up < lower_bound_up.to(device)).float()
        condition2_up = (inp_cumsum_up >= upper_bound_up).float()
        upper_bound_down = self.num_spin_down
        lower_bound_down = (self.num_spin_down - (N - torch.arange(1, N+1)))
        condition1_down = (inp_cumsum_down < lower_bound_down.to(device)).float()
        condition2_down = (inp_cumsum_down >= upper_bound_down).float()
        idx = torch.sort(self.shell_order)[1]
        # first entry must be down
        log_psi_cond[:,:,0] = log_psi_cond[:,:,0].masked_fill(condition1_up[:,idx]==1, float('-inf'))
        log_psi_cond[:,:,2] = log_psi_cond[:,:,2].masked_fill(condition1_up[:,idx]==1, float('-inf'))
        # second entry must be down
        log_psi_cond[:,:,0] = log_psi_cond[:,:,0].masked_fill(condition1_down[:,idx]==1, float('-inf'))
        log_psi_cond[:,:,1] = log_psi_cond[:,:,1].masked_fill(condition1_down[:,idx]==1, float('-inf'))
        # first entry must be up
        log_psi_cond[:,:,1] = log_psi_cond[:,:,1].masked_fill(condition2_up[:,idx]==1, float('-inf'))
        log_psi_cond[:,:,3] = log_psi_cond[:,:,3].masked_fill(condition2_up[:,idx]==1, float('-inf'))
        # second entry must be up
        log_psi_cond[:,:,2] = log_psi_cond[:,:,2].masked_fill(condition2_down[:,idx]==1, float('-inf'))
        log_psi_cond[:,:,3] = log_psi_cond[:,:,3].masked_fill(condition2_down[:,idx]==1, float('-inf'))
        # # first entry must be down
        # log_psi_cond[:,:,0].masked_fill(condition1_up[:,idx]==1, float('-inf'))
        # log_psi_cond[:,:,2].masked_fill(condition1_up[:,idx]==1, float('-inf'))
        # # second entry must be down
        # log_psi_cond[:,:,0].masked_fill(condition1_down[:,idx]==1, float('-inf'))
        # log_psi_cond[:,:,1].masked_fill(condition1_down[:,idx]==1, float('-inf'))
        # # first entry must be up
        # log_psi_cond[:,:,1].masked_fill(condition2_up[:,idx]==1, float('-inf'))
        # log_psi_cond[:,:,3].masked_fill(condition2_up[:,idx]==1, float('-inf'))
        # # second entry must be up
        # log_psi_cond[:,:,2].masked_fill(condition2_down[:,idx]==1, float('-inf'))
        # log_psi_cond[:,:,3].masked_fill(condition2_down[:,idx]==1, float('-inf'))
        # # first entry must be down
        # log_psi_cond[:,:,0].masked_fill_(condition1_up[:,idx]==1, float('-inf'))
        # log_psi_cond[:,:,2].masked_fill_(condition1_up[:,idx]==1, float('-inf'))
        # # second entry must be down
        # log_psi_cond[:,:,0].masked_fill_(condition1_down[:,idx]==1, float('-inf'))
        # log_psi_cond[:,:,1].masked_fill_(condition1_down[:,idx]==1, float('-inf'))
        # # first entry must be up
        # log_psi_cond[:,:,1].masked_fill_(condition2_up[:,idx]==1, float('-inf'))
        # log_psi_cond[:,:,3].masked_fill_(condition2_up[:,idx]==1, float('-inf'))
        # # second entry must be up
        # log_psi_cond[:,:,2].masked_fill_(condition2_down[:,idx]==1, float('-inf'))
        # log_psi_cond[:,:,3].masked_fill_(condition2_down[:,idx]==1, float('-inf'))
        # inf_sum = (log_psi_cond == float('-inf')).sum()
        # print(f"inf_sum: {inf_sum}")
        return log_psi_cond

    @torch.no_grad()
    def sample(self, bs, num_samples):
        self.sampling = True
        sample_multinomial = True
        # random initialize a configuration of values +- 1
        uniq_samples = (torch.randn(1, self.num_sites).to(self.device) > 0.0).float() * 2 - 1
        uniq_count = torch.tensor([num_samples]).to(self.device)
        for i in self.shell_order:
            prob = self.forward(uniq_samples)[:, i] # num_uniq, 4
            num_uniq = uniq_samples.shape[0]
            uniq_samples = uniq_samples.repeat(4,1) # 4*num_uniq, num_sites
            # convert [|-1,-1>, |1,-1>, |-1,1>, |1,1>] to [0, 1, 2, 3]
            uniq_samples[:num_uniq, 2*i] = -1
            uniq_samples[:num_uniq, 2*i+1] = -1
            uniq_samples[num_uniq:2*num_uniq, 2*i] = 1
            uniq_samples[num_uniq:2*num_uniq, 2*i+1] = -1
            uniq_samples[2*num_uniq:3*num_uniq, 2*i] = -1
            uniq_samples[2*num_uniq:3*num_uniq, 2*i+1] = 1
            uniq_samples[3*num_uniq:4*num_uniq, 2*i] = 1
            uniq_samples[3*num_uniq:4*num_uniq, 2*i+1] = 1
            if sample_multinomial:
                uniq_count = torch.tensor(multinomial_arr(uniq_count.long().data.cpu().numpy(), prob.data.cpu().numpy())).T.flatten().to(prob.device)
            else:
                uniq_count = (uniq_count.unsqueeze(-1)*prob).T.flatten().round()
            keep_idx = uniq_count > 1
            uniq_samples = uniq_samples[keep_idx]
            uniq_count = uniq_count[keep_idx]
            uniq_samples = uniq_samples[uniq_count.sort()[1][-2*bs:]]
            uniq_count = uniq_count[uniq_count.sort()[1][-2*bs:]]
        uniq_samples = uniq_samples[uniq_count.sort()[1][-bs:]]
        uniq_count = uniq_count[uniq_count.sort()[1][-bs:]]
        self.sampling = False
        return uniq_samples, uniq_count