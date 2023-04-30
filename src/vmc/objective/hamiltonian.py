import torch
import torch.nn as nn
from src.complex import scalar_mult, real, imag



class Hamiltonian(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, config):
        # TODO: Implement forward pass for Hamiltonian
        pass


class Energy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, log_psi, local_energies, weight=None):
        # Save local energies and compute mean energy
        ctx.local_energies = local_energies
        if weight is None:
            ctx.mean_energy = torch.mean(local_energies, dim=0)
        else:
            ctx.mean_energy = torch.sum(local_energies * weight.unsqueeze(-1), dim=0)
        # Return local energies
        return local_energies

    @staticmethod
    def backward(ctx, log_psi_grad):
        # Compute gradient
        grad = ctx.local_energies - ctx.mean_energy
        grad_total = 2 * scalar_mult(log_psi_grad, grad)
        # Return gradient and None for other arguments
        return grad_total, None, None
