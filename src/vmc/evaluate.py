import numpy as np
import torch

from src.complex import real

def test(model, sampler, hamiltonian, batch_size, num_samples, inner_iter, world_size):
    device = list(model.parameters())[0].device
    model.eval()
    with torch.no_grad():
        sampler.output_stats = True
        samples, avg_accept = sampler(batch_size, num_samples)
        sampler.output_stats = False
        if isinstance(samples, list):
            samples, count = samples
            weight = count / count.sum()
        else:
            weight = 1 / samples.shape[0] * torch.ones([samples.shape[0]])
        if isinstance(samples, (np.ndarray, np.generic)):
            samples = torch.tensor(samples).float().to(device)
        # num_uniq = samples.shape[0]
        num_uniq = torch.unique(samples, dim=0).shape[0]
        bs = np.ceil(samples.shape[0] / world_size / inner_iter).astype(np.int64)
        scores = torch.tensor([]).to(samples.device)
        if bs == 1:
            inner_iter = np.ceil(samples.shape[0] / world_size).astype(np.int64)
        for i in range(world_size*inner_iter):
            score, _ = hamiltonian.compute_local_energy(samples[i*bs:(i+1)*bs], model)
            scores = torch.cat((scores, score.float()), dim=0)
        weight = weight.to(scores.device)
        mean = (scores * weight.unsqueeze(-1)).sum(0)
        score = real(mean)
        std = ((real(scores) - score)**2 * weight.unsqueeze(-1)).sum().sqrt()
        max_score = real(scores).min()
    return mean, score.item(), std.item(), max_score.item(), avg_accept, num_uniq
