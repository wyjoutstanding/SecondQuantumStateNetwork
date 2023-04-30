from ast import excepthandler
import sys
import torch
import random
import numpy as np


from torch.distributions.bernoulli import Bernoulli


class FLOWSampler(torch.nn.Module):
    def __init__(self, model, batch_size, state_size, total_samples=1e6):
        super(FLOWSampler, self).__init__()
        self.model = model
        self.total_samples = total_samples
        self.batch_size = batch_size
        self.state_size = state_size
        self.output_stats = False

    @torch.no_grad()
    def forward(self, bs=None, num_samples=None):
        if 'NADE' in str(type(self.model)):
            states, counts, _, _ = self.model.sample(self.total_samples)
            uniq_samples = states.to(self.model.device)
            uniq_count = counts.to(self.model.device)
            num_uniqs = len(uniq_count)
            if num_uniqs > self.batch_size:
                sorted_idx = uniq_count.sort()[1]
                uniq_samples_1 = uniq_samples[sorted_idx[-self.batch_size//2:]]
                uniq_count_1 = uniq_count[sorted_idx[-self.batch_size//2:]]
                num_left = len(uniq_count) - self.batch_size // 2
                idxs = random.sample(range(num_left), self.batch_size // 2)
                uniq_samples_2 = uniq_samples[sorted_idx[idxs]]
                uniq_count_2 = uniq_count[sorted_idx[idxs]]
                uniq_samples = torch.cat((uniq_samples_1, uniq_samples_2), dim=0)
                uniq_count = torch.cat((uniq_count_1, uniq_count_2), dim=0)
        else:
            self.model.eval()
            uniq_samples, uniq_count = self.model.sample(bs, num_samples)
        if self.output_stats:
            return [uniq_samples, uniq_count], 0.0
        else:
            return [uniq_samples, uniq_count]
    
