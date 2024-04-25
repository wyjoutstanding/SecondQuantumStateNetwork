import torch
import torch.nn as nn
import torch.nn.functional as F
from backpack import backpack
from backpack.extensions import BatchGrad
from src.vmc.objective.hamiltonian import Energy

class Base(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.energy = Energy.apply
        self.sampler_type = 'MCMC'

    def forward(self, configuration):
        pass

    def log_dev(self, log_psi, model, imag=False):
        # import pdb; pdb.set_trace()
        device = log_psi.device
        bs, log_psi_shape = log_psi.shape[0], log_psi.shape[-1]
        log_psi_real_or_imag = log_psi[:, imag].sum()
        with backpack(BatchGrad()):
            log_psi_real_or_imag.backward(retain_graph=True)
        log_psi_grads_flatten = torch.cat([param.grad_batch.reshape(bs, -1) for name, param in model.named_parameters() if param.grad is not None], dim=-1).to(device)
        model.zero_grad()
        return log_psi_grads_flatten

    def log_dev_real(self, log_psi, model):
        return self.log_dev(log_psi, model, imag=False)

    def log_dev_imag(self, log_psi, model):
        return self.log_dev(log_psi, model, imag=True)

    def compute_loss(self, log_psi, local_energies):
        return self.energy(log_psi, local_energies)

    def set_mask_device(self):
        pass
