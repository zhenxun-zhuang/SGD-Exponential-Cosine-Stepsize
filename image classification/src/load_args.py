"""
Command-line argument parsing.
"""

import argparse

def load_args():
    parser = argparse.ArgumentParser(description='Comparing different SGD variants')

    parser.add_argument('--optim-method', type=str, default='SGD',
                        help='Which optimizer to use.')
    parser.add_argument('--eta0', type=float, default=0.1,
                        help='Initial learning rate (default: 0.1).')
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='Decay factor (default: 0.1).')
    parser.add_argument('--c', type=float, default=0.1,
                        help='Line search parameter (default: 0.1).')
    parser.add_argument('--nesterov', action='store_true',
                        help='Use nesterov momentum (default: False).')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum used in optimizer (default: 0.9).')    
    parser.add_argument('--weight-decay', type=float, default=0.0005,
                        help='Weight decay used in optimizer (default: 0.0005).')
    parser.add_argument('--milestones', nargs='*', default=[],
                        help='Used for SGD stagewise decay denoting when to decrease the step size, unit in iteration (default: []).')   
    parser.add_argument('--patience', type=int, default=10,
                        help='Used in ReduceLROnPlateau denoting number of epochs with no improvement after which learning rate will be reduced (default: 10).')
    parser.add_argument('--threshold', type=float, default=1e-4,
                        help='Used in ReduceLROnPlateau for measuring the new optimum, to only focus on significant changes (default: 1e-4).')

    parser.add_argument('--train-epochs', type=int, default=100,
                        help='Number of train epochs (default: 100).')
    parser.add_argument('--batchsize', type=int, default=16,
                        help='How many images in each train epoch (default: 16).')
    parser.add_argument('--validation', action='store_true',
                        help='Do validation (True) or test (False) (default: False).')        
    parser.add_argument('--val-ratio', type=float, default=0.1,
                        help='Percentage of training samples used as validation (default: 0.1).')
    parser.add_argument('--eval-interval', type=int, default=10,
                        help='How often should the model be evaluated during training, unit in epochs (default: 10).')

    parser.add_argument('--dataset', type=str, default='CIFAR10',
                        choices=['MNIST', 'FashionMNIST', 'CIFAR10', 'CIFAR100'],
                        help='Which dataset to run on (default: CIFAR10).')  
    parser.add_argument('--dataroot', type=str, default='../data',
                        help='Where to retrieve data (default: ../data).')        
    parser.add_argument('--use-cuda', action='store_true',
                        help='Use CUDA (default: False).')
    parser.add_argument('--reproducible', action='store_true',
                        help='Ensure reproducibility (default: False).')
    parser.add_argument('--seed', type=int, default=0, metavar='S',
                        help='Random seed (default: 0).')

    parser.add_argument('--log-folder', type=str, default='../logs',
                        help='Where to store results.')

    return parser.parse_args()
