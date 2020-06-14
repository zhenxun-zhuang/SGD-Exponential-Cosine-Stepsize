"""
Create train, valid, test iterators for a chosen dataset.
"""

import random
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader

def data_loader(dataset_name, dataroot, batch_size, val_ratio):
    """
    Args:
        dataset_name (str): the name of the dataset to use, currently only
            supports 'MNIST', 'FashionMNIST', 'CIFAR10' and 'CIFAR100'.
        dataroor (str): the location to save the dataset.
        batch_size (int): batch size used in training.
        val_ratio (float): the percentage of trainng data used as validation.

    Outputs:
        iterators over training, validation, and test data.
    """
    if ((val_ratio < 0) or (val_ratio > 1.0)):
        raise ValueError("[!] val_ratio should be in the range [0, 1].")    

    test_batchsize = 100

    # Mean and std are obtained for each channel from all training images.
    if dataset_name == 'CIFAR10':
        dataset = torchvision.datasets.CIFAR10
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                         (0.2470, 0.2435, 0.2616))
    elif dataset_name == 'CIFAR100':
        dataset = torchvision.datasets.CIFAR100
        normalize = transforms.Normalize((0.5071, 0.4866, 0.4409),
                                         (0.2673, 0.2564, 0.2762))
    elif dataset_name == 'MNIST':
        dataset = torchvision.datasets.MNIST
        normalize = transforms.Normalize((0.1307,), (0.3081,))
    elif dataset_name == 'FashionMNIST':
        dataset = torchvision.datasets.FashionMNIST
        normalize = transforms.Normalize((0.2860,), (0.3530,))

    if dataset_name.startswith('CIFAR'):
        # Follows Lee et al. Deeply supervised nets. 2014.
        transform_train = transforms.Compose([transforms.RandomHorizontalFlip(),
                                              transforms.RandomCrop(32, padding=4),
                                              transforms.ToTensor(),
                                              normalize])
        transform_test = transforms.Compose([transforms.ToTensor(),
                                             normalize])
    elif dataset_name in ['MNIST', 'FashionMNIST']:
        transform_train = transforms.Compose([transforms.ToTensor(),
                                              normalize])
        transform_test = transform_train

    # load and split the train dataset.
    train_set = dataset(root=dataroot, train=True,
                        download=True, transform=transform_train)
    val_set = dataset(root=dataroot, train=True,
                      download=True, transform=transform_test)

    num_train = len(train_set)
    indices = list(range(num_train))
    random.shuffle(indices)
    split = int(val_ratio * num_train)
    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    train_loader = DataLoader(train_set, batch_size=batch_size,
                              sampler=train_sampler,
                              num_workers=1, pin_memory=True)

    valid_sampler = SubsetRandomSampler(valid_idx)
    valid_loader = DataLoader(val_set, batch_size=test_batchsize,
                              sampler=valid_sampler,
                              num_workers=1, pin_memory=True)

    # Load the test dataset.
    test_set = dataset(root=dataroot, train=False,
                       download=True, transform=transform_test)
    test_loader = DataLoader(test_set, batch_size=test_batchsize,
                             shuffle=False, num_workers=1,
                             pin_memory=True)

    return (train_loader, valid_loader, test_loader)   
 