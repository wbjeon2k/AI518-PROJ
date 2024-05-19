import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision.datasets import CIFAR10,CIFAR100,MNIST,FashionMNIST
import torchvision.transforms as TF
import yaml

def load_config_from_yaml():
    with open('config.yaml', 'r') as f:
        loaded_config = yaml.load(f, Loader=yaml.SafeLoader)
    return loaded_config

def get_total_number_of_cls(dset_name):

    if dset_name == 'cifar10':
        return 10
    elif dset_name == 'cifar100':
        return 100
    elif dset_name == 'mnist':
        return 10
    elif dset_name == 'fmnist':
        return 10
    else:
        raise NotImplementedError('Currently unavailable dataset')
    
    
def get_dataset(dataset_name, dataset_path, transform: TF, is_train = True, ):
    if dataset_name is None:
        ret = CIFAR10(root=dataset_path, train=is_train, transform=transform)
        return ret
    
    dataset_name = dataset_name.lower()
    
    if dataset_name == 'cifar10':
        return CIFAR10(root=dataset_path, train=is_train, transform=transform)
    elif dataset_name == 'cifar100':
        return CIFAR100(root=dataset_path, train=is_train, transform=transform)
    elif dataset_name == 'mnist':
        return MNIST(root=dataset_path, train=is_train, transform=transform)
    elif dataset_name == 'fmnist':
        return FashionMNIST(root=dataset_path, train=is_train, transform=transform)
    else:
        raise NotImplementedError('Currently unavailable dataset')

def parse_train_config(config_total):
    
    train_configs = config_total['train']
    dset_name = train_configs['dataset_name']
    dset_path = train_configs['dataset_path']
    optim_name = train_configs['optim_name']
    epoch_base = train_configs['epoch_base']
    epoch_per_task = train_configs['epoch_per_task']
    # = train_configs['']
    
    return dset_name, dset_path, optim_name, epoch_base, epoch_per_task