import re
import logging
import numpy as np
import torch
from pytorch_model_summary import summary
from torch.autograd import Variable



def get_model(model_name: str, device: str, print_model_info: bool, **kwargs) -> torch.nn.Module:
    """
    Get a PyTorch model based on the given model name and arguments.
    Args:
    - model_name: Name of the model to be loaded.
    - device: Device to be used for model inference and training.
    - print_model_info: Whether to print the model summary.
    - **kwargs: Other arguments specific to the model.
    Returns:
    - model: The loaded PyTorch model.
    """
    if model_name == 'naqs':
        from .naqs import NADE
        model = NADE(**kwargs)
    elif model_name == 'made':
        from .made import MADE
        model = MADE(**kwargs)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")
    if print_model_info:
        print(summary(model, torch.zeros(10, model.num_sites), show_input=False))
    model.eval()
    model.device = device
    return model.to(device)


def load_model(model: torch.nn.Module, model_load_path: str, strict: bool = True) -> torch.nn.Module:
    """
    Load a trained PyTorch model from a given path and update the model parameters.
    Args:
    - model: The PyTorch model to be updated.
    - model_load_path: The path of the saved model file.
    - strict: Whether to strictly enforce that the keys in state_dict match the keys returned by model.state_dict().
    Returns:
    - model: The updated PyTorch model.
    """
    # Load the saved model from the file path
    bad_state_dict = torch.load(model_load_path, map_location='cpu')
    # Rename the keys in state_dict from 'module.' to '' (for loading into a non-DistributedDataParallel model)
    correct_state_dict = {re.sub(r'^module\.', '', k): v for k, v in bad_state_dict.items()}
    # If strict is False, only load the parameters that have the same shape as the corresponding parameters in the model
    if not strict:
        logging.info(f"Loading {len(correct_state_dict)} params")
        own_state = model.state_dict()
        final_state_dict = {}
        for name, param in correct_state_dict.items():
            if name not in own_state:
                continue
            param = param.data
            own_param = own_state[name].data
            if own_param.shape == param.shape:
                final_state_dict[name] = param
        correct_state_dict = final_state_dict
        logging.info(f"Loaded {len(correct_state_dict)} params")
    # Load the state_dict into the model
    model.load_state_dict(correct_state_dict, strict=strict)
    model.eval()
    model.zero_grad()
    return model


def save_model(model, model_save_path):
    logging.info("[*] Save model to {}...".format(model_save_path))
    torch.save(model.state_dict(), model_save_path)
    return model_save_path


def get_model_gradient(model):
    grads = []
    for param in model.parameters():
        grads.append(param.grad.view(-1))
    grads = torch.cat(grads)
    return grads


def get_input_degrees(model, nin, nout):
    # Use a seed for reproducibility
    rng = np.random.RandomState(seed=14)
    # Generate a random binary input of size (1, nin)
    x = (rng.rand(1, nin) > 0.5).astype(np.float32)
    input_degrees = []
    # Iterate over the output nodes to determine the input degrees
    for k in range(nout):
        # Convert the input to a tensor with requires_grad=True to compute gradients
        xtr = torch.from_numpy(x).requires_grad_(True)
        # Pass the input through the model to obtain the output
        xtrhat = model(xtr.to(next(model.parameters()).device))
        # Compute the loss as the k-th output value
        loss = xtrhat[0, k]
        # Compute the gradients of the loss with respect to the input
        loss.backward()
        # Determine which inputs the output depends on
        depends = (xtr.grad[0].numpy() != 0).astype(np.uint8)
        depends_ix = list(np.where(depends)[0])
        # Check if the output depends on an odd number of inputs
        isok = k % 2 not in depends_ix
        # Print the output dependencies and whether it satisfies the oddness constraint
        print(f"Output {k:2d} depends on inputs: {depends_ix:30s} : {'OK' if isok else 'NOT OK'}")
        # Store the number of input dependencies for each output
        input_degrees.append(len(depends_ix))
    # Sort the input degrees in increasing order and return the indices corresponding to the first nin values
    input_degrees = np.array(input_degrees)
    input_degrees_sorted = np.sort(input_degrees)[:nin]
    input_degrees_indices = np.argsort(input_degrees)[:nin]
    return torch.tensor(input_degrees_indices)


def multinomial_arr(count, p):
    # Copy the count array to avoid modifying it in place
    count = np.copy(count)
    out = np.zeros_like(p, dtype=int)
    # Compute the cumulative sums of the probabilities
    ps = np.cumsum(p, axis=-1)
    # Avoid division by zero and NaNs by setting the probabilities to zero where the cumulative sum is zero
    condp = np.divide(p, ps, out=np.zeros_like(p), where=ps != 0)
    # Iterate over the columns of p in reverse order
    for i in range(p.shape[-1] - 1, 0, -1):
        # Sample from a binomial distribution using the conditional probabilities
        binsample = np.random.binomial(count, condp[..., i])
        # Update the output array and the count array
        out[..., i] = binsample
        count -= binsample
    # Assign the remaining count to the first column of the output array
    out[..., 0] = count
    return out
