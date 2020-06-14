import torch.optim

from .dfw import DFW
from .bpgrad import BPGrad
from .sgd_lr_decay import SGDLRDecay

def get_optimizer(args, parameters, T_max):
    """
    Available optimizers:
    - SGD
    - Adam
    - Adagrad
    - AMSGrad
    - DFW
    - BPGrad
    - Cosine decay
    - Exponential decay
    - O(1/t) decay
    - O(1/sqrt{t}) decay
    """
    if args.opt == 'sgd':
        optimizer = torch.optim.SGD(parameters, lr=args.eta, weight_decay=args.l2,
                                    momentum=args.momentum, nesterov=bool(args.momentum))
    elif args.opt == "adam":
        optimizer = torch.optim.Adam(parameters, lr=args.eta, weight_decay=args.l2)
    elif args.opt == "adagrad":
        optimizer = torch.optim.Adagrad(parameters, lr=args.eta, weight_decay=args.l2)
    elif args.opt == "amsgrad":
        optimizer = torch.optim.Adam(parameters, lr=args.eta, weight_decay=args.l2, amsgrad=True)
    elif args.opt == 'dfw':
        optimizer = DFW(parameters, eta=args.eta, momentum=args.momentum, weight_decay=args.l2)
    elif args.opt == 'bpgrad':
        optimizer = BPGrad(parameters, eta=args.eta, momentum=args.momentum, weight_decay=args.l2)
    elif args.opt.startswith('SGD') and args.opt.endswith('Decay'):
        if args.opt == 'SGD_Exp_Decay':
            scheme = 'exp'
        elif args.opt == 'SGD_1t_Decay':
            scheme = '1t'
        elif args.opt == 'SGD_1sqrt_Decay':
            scheme = '1sqrt'
        elif args.opt == 'SGD_Cosine_Decay':
            scheme = 'cosine'
        optimizer = SGDLRDecay(params=parameters, scheme=scheme, eta0=args.eta,
                               alpha=args.alpha, T_max=T_max, momentum=args.momentum,
                               weight_decay=args.l2, nesterov=bool(args.momentum))
    else:
        raise ValueError(args.opt)

    print("Optimizer: \t {}".format(args.opt.upper()))

    optimizer.gamma = 1
    optimizer.eta = args.eta

    return optimizer


def decay_optimizer(optimizer, decay_factor=0.1):
    if isinstance(optimizer, torch.optim.SGD):
        for param_group in optimizer.param_groups:
            param_group['lr'] *= decay_factor
        # update state
        optimizer.eta = optimizer.param_groups[0]['lr']
    else:
        raise ValueError
