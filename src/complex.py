import numpy as np
import torch



# Simplify the functions by using torch's built-in indexing for the last dimension
def real(x):
    return x[..., 0]


def imag(x):
    return x[..., 1]


def exp(x):
    amp, phase = real(x).exp(), imag(x)
    return torch.stack([amp * phase.cos(), amp * phase.sin()], -1)

def torch_to_numpy(x):
    return np.array(real(x).numpy()+1j*imag(x).numpy())

def np_to_torch(x):
    #return torch.FloatTensor(np.stack([np.real(x), np.imag(x)], -1))
    return torch.tensor(np.stack([np.real(x), np.imag(x)], -1))

# Simplify the code by making sure that y is a torch tensor before converting it to x's type
def scalar_mult(x, y):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if isinstance(y, np.ndarray):
        y = torch.from_numpy(y)
    y = y.to(x.device)
    re = real(x) * real(y) - imag(x) * imag(y)
    im = real(x) * imag(y) + imag(x) * real(y)
    return torch.stack([re, im], dim=-1) if torch.is_tensor(x) else np.stack([re, im], axis=-1)
