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

def get_train_set_config(config_total):
    
    train_configs = config_total['train']
    dset_name = train_configs['dataset_name']
    dset_path = train_configs['dataset_path']
    
    return dset_name, dset_path

def get_train_hyper_parameters(config_total):
    hparams = config_total['hparam']
    
    optim_name = hparams['optim_name']
    learning_rate = hparams['learning_rate']
    epoch_base = hparams['epoch_base']
    epoch_per_task = hparams['epoch_per_task']
    batch_size = hparams['batch_size']
    
    return optim_name, learning_rate, epoch_base, \
        epoch_per_task, batch_size
        
def get_task_info(config_total):
    task_info = config_total['task_info']
    
    cls_total = task_info['cls_total']
    cls_base = task_info['cls_base']
    num_tasks = task_info['num_tasks']
    cls_per_task = task_info['cls_per_task']
    
    return cls_total, cls_base, num_tasks, cls_per_task

def get_optim(optim_name, MODEL : nn.Module, lr):
    optim_name = optim_name.lower()
    if optim_name == 'adam':
        return optim.Adam(MODEL.parameters(), lr=lr)
    elif optim_name == 'sgd':
        return optim.SGD(MODEL.parameters(), lr=lr)
    else:
        raise NotImplementedError('Currently not available')