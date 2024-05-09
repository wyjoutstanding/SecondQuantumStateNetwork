import torch
import torch.nn as nn
import numpy as np
import itertools

import src.complex as cplx

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
        grad_total = 2 * cplx.scalar_mult(log_psi_grad, grad)
        # Return gradient and None for other arguments
        return grad_total, None, None

class MolecularHamiltoian(Hamiltonian):
    def __init__(self, operators, coefficients):
        self.num_terms, self.input_dim = operators.shape
        print("Number of terms is {}.".format(self.num_terms))
        assert coefficients.shape[0] == self.num_terms
        # product of identity operators by default, encoded as 0
        operators = torch.tensor(operators)
        self.coefficients = torch.tensor(coefficients)
        self.coefficients = torch.stack((self.coefficients.real, self.coefficients.imag), dim=-1)
        # find index of pauli X,Y,Z operators
        pauli_x_idx = (operators==1).int() # [num_terms, input_dim]
        pauli_y_idx = (operators==2).int() # [num_terms, input_dim]
        pauli_z_idx = (operators==3).int() # [num_terms, input_dim]
        # track the exponential of -i
        self.num_pauli_y = pauli_y_idx.sum(-1) # [num_terms]
        # the unique element has flipped value if the corresponding pauli is x or y.
        flip_idx = pauli_x_idx + pauli_y_idx # [num_terms, input_dim]
        # self.flip_idx = flip_idx
        del pauli_x_idx
        # only the entry value with y or z pauli is multiplied
        self.select_idx = pauli_y_idx + pauli_z_idx
        del pauli_y_idx
        del pauli_z_idx
        unique_flips, unique_indices = np.unique(np.array(flip_idx), axis=0, return_inverse=True)
        self.unique_flips = torch.tensor(unique_flips)
        self.unique_indices = torch.tensor(unique_indices)
        self.unique_num_terms = self.unique_flips.shape[0]

    def compute_local_energy(self, samples, model):
        # see appendix B of https://arxiv.org/pdf/1909.12852.pdf
        # x [bs, input_dim]
        # determine the unique element
        x = samples
        bs = x.shape[0]
        x_k = x.unsqueeze(1) * (self.unique_flips.unsqueeze(0)*(-2) + 1) # [bs, unique_num_terms, input_dim]
        # forward pass
        # log_psi_k = model(x_k.reshape(-1, self.input_dim)) # [bs*unique_num_terms, 2]
        # log_psi = model(x) # [bs, 2]
        output = model(torch.cat((x_k.reshape(-1, self.input_dim), x)))
        log_psi_k, log_psi = output[:-bs], output[-bs:]
        # print(f"log_psi_k: {log_psi_k}")
        if len(log_psi.shape) == 1: # if not complex
            log_psi_k = torch.stack([log_psi_k, torch.zeros_like(log_psi_k).to(log_psi_k.device)], dim=-1)
            log_psi = torch.stack([log_psi, torch.zeros_like(log_psi).to(log_psi.device)], dim=-1)
        log_psi_k = log_psi_k.reshape(bs, self.unique_num_terms, 2) # [bs, unique_num_terms, 2]
        log_psi_k = log_psi_k[:, self.unique_indices] # [bs, num_terms, 2]
        ratio = cplx.exp(log_psi_k-log_psi.unsqueeze(1)) # [bs, num_terms, 2]
        # compute matrix element
        # Eq. B3
        part2 = (x.unsqueeze(1).repeat(1, self.num_terms, 1) * self.select_idx.unsqueeze(0) + (1-self.select_idx).unsqueeze(0)).prod(-1) # [bs, num_terms, input_dim]
        part2 = torch.stack((part2, torch.zeros_like(part2)), dim=-1)
        part1 = (1j)**self.num_pauli_y.detach().cpu().numpy()
        part1 = torch.stack((torch.tensor(part1.real), torch.tensor(part1.imag)), dim=-1).float().to(x.device)
        mtx_k = cplx.scalar_mult(part1, part2) # [bs, num_terms, 2]
        # total local energy
        local_energy = cplx.scalar_mult(self.coefficients.unsqueeze(0), cplx.scalar_mult(mtx_k, ratio)).sum(1) # [bs, 2]
        return local_energy, log_psi

    def set_device(self, device):
        self.coefficients = self.coefficients.to(device)
        self.num_pauli_y = self.num_pauli_y.to(device)
        self.select_idx = self.select_idx.to(device)
        self.unique_flips = self.unique_flips.to(device)
        self.unique_indices = self.unique_indices.to(device)


def laplacian_to_mtx(laplacian):
    size = laplacian.shape[0]
    coef = []
    mtx = []
    for i in range(size):
        for j in range(size):
            if i == j:
                coef.append(-laplacian[i,j]/4)
                ops = np.zeros(size)
                ops[i] = 0
                mtx.append(ops)
            elif laplacian[i,j] != 0:
                coef.append(-laplacian[i,j]/4)
                ops = np.zeros(size)
                ops[i] = 3
                ops[j] = 3
                mtx.append(ops)
    return np.array(mtx), np.array(coef)

def tim_to_mtx(h, g, gg):
    size = gg.shape[0]
    coef = []
    mtx = []
    for i in range(size):
        for j in range(size):
            coef.append(gg[i,j])
            ops = np.zeros(size)
            ops[i] = 3
            ops[j] = 3
            mtx.append(ops)
    for i in range(size):
        coef.append(g[i])
        ops = np.zeros(size)
        ops[i] = 3
        mtx.append(ops)
    for i in range(size):
        coef.append(h[i])
        ops = np.zeros(size)
        ops[i] = 1
        mtx.append(ops)
    return np.array(mtx), np.array(coef)

def parse_hamiltonian_string(hamiltonian_string, num_sites, **kwargs):
    splitted_string = hamiltonian_string.split('+\n')
    num_terms = len(splitted_string)
    params = np.zeros([num_terms]).astype(np.complex128)
    hmtn_ops = np.zeros([num_terms, num_sites])
    for i,term in enumerate(splitted_string):
        params[i] = complex(term.split(' ')[0])
        ops = term[term.index('[')+1:term.index(']')]
        ops_lst = ops.split(' ')
        for op in ops_lst:
            if op == '':
                continue
            pauli_type = op[0]
            idx = int(op[1:])
            if pauli_type == 'X':
                encoding = 1
            elif pauli_type == 'Y':
                encoding = 2
            elif pauli_type == 'Z':
                encoding = 3
            elif pauli_type == 'I':
                encoding = 0
            else:
                raise "Unknown pauli_type!"
            hmtn_ops[i, idx] = encoding
    return hmtn_ops, params


def dense_hamiltonian(num_sites, hmtn_ops, coefs, unique_flips, select_idx, num_pauli_y):
    inputs = torch.tensor(np.array(list(itertools.product([0, 1], repeat=num_sites)))) * 2.0 - 1.0
    num_terms = coefs.shape[0]
    assert num_terms == hmtn_ops.shape[0]
    assert num_sites == hmtn_ops.shape[1]
    size = inputs.shape[0]
    mtx = np.zeros((size, size), dtype=np.complex128)
    pauli_x_idx = (hmtn_ops==1).int()
    pauli_y_idx = (hmtn_ops==2).int()
    pauli_z_idx = (hmtn_ops==3).int()
    for i in range(size):
        print(i)
        x = inputs[i]
        for j in range(size):
            xx = inputs[j]
            num_pauli_y = pauli_y_idx.sum(-1)
            flip_idx = pauli_x_idx + pauli_y_idx
            select_idx = pauli_y_idx + pauli_z_idx
            cond = ((x.unsqueeze(0) * (flip_idx*(-2)+1)) == xx.unsqueeze(0)).prod(-1).detach().cpu().numpy()
            xx_prod = x.unsqueeze(0).repeat(num_terms, 1) * select_idx
            xx_prod += 0.9 # for efficiency, we only care about the sign here
            val = (1j)**num_pauli_y.detach().cpu().numpy() * xx_prod.prod(-1).sign().detach().cpu().numpy()
            mtx[i,j] = (cond * val * coefs.detach().cpu().numpy()).sum()
    return mtx



if __name__ == '__main__':
    pass