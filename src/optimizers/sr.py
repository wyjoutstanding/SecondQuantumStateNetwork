import numpy as np
import torch
from scipy.sparse.linalg import minres, LinearOperator

from .util import apply_grad, vec_to_grad


class SR:
    def __init__(self, damping_factor):
        self.flatten_delta_centred = None
        self.damping_factor = damping_factor

    def update_delta(self, flatten_log_grads_batch, weight):
        # calculate mean of flatten_log_grads_batch
        flatten_delta_mean = np.sum(flatten_log_grads_batch * weight, axis=0, keepdims=True)
        # calculate centred log gradients
        self.flatten_delta_centred = flatten_log_grads_batch - flatten_delta_mean 

    def apply_sr_mtx(self, x):
        # calculate the Hessian vector product
        flatten_delta_centred = self.flatten_delta_centred * np.sqrt(self.weight)
        flatten_delta_centred_real = flatten_delta_centred[:, :, 0]
        flatten_delta_centred_imag = flatten_delta_centred[:, :, 1]
        Ax = flatten_delta_centred_real.T @ (flatten_delta_centred_real @ np.expand_dims(x, -1)) + flatten_delta_centred_imag.T @ (flatten_delta_centred_imag @ np.expand_dims(x, -1))
        Ax = np.squeeze(Ax)
        if self.damping_factor > 0.0:
            Ax = Ax + self.damping_factor * x
        return Ax

    def compute_grad(self, b):
        # calculate the natural gradient using sparse linear solver
        param_size = b.shape[0]
        A = LinearOperator((param_size, param_size), matvec=self.apply_sr_mtx)
        x = minres(A, b, x0=b)[0]
        return x

    @torch.no_grad()
    def compute_natural_grad(self, grad, flatten_log_grads_batch, weight):
        # compute natural gradient
        device = grad.device
        grad = grad.cpu().numpy()
        self.weight = weight.unsqueeze(-1).unsqueeze(-1).cpu().numpy()
        flatten_log_grads_batch = flatten_log_grads_batch.cpu().numpy()
        self.update_delta(flatten_log_grads_batch, weight.cpu().numpy())
        natural_grad = self.compute_grad(grad)
        natural_grad = torch.from_numpy(natural_grad).to(device)
        # clip the gradient to avoid gradient blow
        natural_grad = torch.clamp(natural_grad, -100, 100)
        return natural_grad

    @torch.no_grad()
    def apply_sr_grad(self, model, grad, flatten_log_grads_batch, weight):
        # apply stochastic reconfiguration gradient
        natural_grad = self.compute_natural_grad(grad, flatten_log_grads_batch, weight)
        apply_grad(model, vec_to_grad(natural_grad, model))
