import torch.optim as optim


def get_optimizer(opt_name, model, learning_rate, weight_decay=0.0):
    if opt_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay)
    elif opt_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif opt_name == 'adadelta':
        optimizer = optim.Adadelta(model.parameters())
    elif opt_name == 'adamax':
        optimizer = optim.Adamax(model.parameters(), lr=learning_rate)
    elif opt_name == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=learning_rate)
    elif opt_name == 'lbfgs':
        optimizer = optim.LBFGS(model.parameters(), lr=learning_rate)
    elif opt_name == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    else:
        raise "opt_name not recognized."
    return optimizer
