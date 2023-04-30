import torch



def apply_grad(model, grad):
    """
    Applies gradients to the model parameters and returns the norm of the gradients.
    """
    grad_norm = 0
    for p, g in zip(model.parameters(), grad):
        if p.grad is None:
            p.grad = g.to(p.device)
        else:
            p.grad += g.to(p.device)
        grad_norm += torch.sum(g ** 2)
    grad_norm = grad_norm ** 0.5
    return grad_norm.item()


def mix_grad(grad_list):
    """
    Mixes the gradients from multiple batches.
    """
    mixed_grad = []
    for g_list in zip(*grad_list):
        g_list = torch.stack([g_list[i] for i in range(len(g_list))])
        mixed_grad.append(torch.mean(g_list, dim=0))
    return mixed_grad


def vec_to_grad(vec, model):
    """
    Converts a vector to a list of gradients with the same shapes as the model parameters.
    """
    pointer = 0
    res = []
    for param in model.parameters():
        num_param = param.numel()
        res.append(vec[pointer:pointer+num_param].view_as(param).data)
        pointer += num_param
    return res
