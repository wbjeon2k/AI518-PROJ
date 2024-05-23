import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.data import Dataset,DataLoader, Subset
import numpy as np
import abc

def split_dataset_by_cls(DATASET: Dataset, num_cls):
    """
    DATASET: instance of torch.utils.data.Dataset
    num_cls: total number of classes in the dataset
    """
    
    idx = dict()
    for i in range(num_cls):
        idx[i] = []
        
    len_dataset = len(DATASET)
    
    for i in range(len_dataset):
        _,y = DATASET.__getitem__(i)
        cls_idx = int(y)
        idx[cls_idx].append(i)
    
    return idx

def generate_task_by_idx(num_total, num_base, num_tasks, cls_per_task, as_list = False):
    if as_list is True:
        assert isinstance(cls_per_task, list)
        assert isinstance(len(cls_per_task) == num_tasks)
        sum_tasks = sum(cls_per_task)
        assert num_total == num_base + sum_tasks
    else:
        assert num_total == num_base + num_tasks*cls_per_task
        
    ret = dict()
    
    permute = np.random.permutation(num_total)
    cls_base = permute[:num_base].tolist()
    ret['base'] = cls_base
    
    if as_list is True:
        start_idx = num_base
        for i in range(len(cls_per_task)):
            cls_task_i = permute[start_idx:start_idx+cls_per_task[i]].tolist()
            start_idx = start_idx + cls_per_task[i]
            ret[f'task{i}'] = cls_task_i
    else:
        start_idx = num_base
        for i in range(num_tasks):
            cls_task_i = permute[start_idx : start_idx + cls_per_task].tolist()
            start_idx += cls_per_task
            ret[f'task{i}'] = cls_task_i
            
    return ret

def generate_CIL_task(DATASET : Dataset, debug = False, **kwargs):
    """
    :param DATASET: instance of torch.utils.data.Dataset
    :param cls_total: total number of classes in the DATASET
    :param cls_base: number of classes for the base task
    :param num_tasks: number of class-incremental tasks
    :cls_per_task: If int, a constant number of classes per task. if list, tasks are splitted into cls_per_task[i] number of cls
    """
    cls_total = kwargs['cls_total']
    cls_base = kwargs['cls_base']
    num_tasks = kwargs['num_tasks']
    cls_per_task = kwargs['cls_per_task']
    
    if isinstance(cls_per_task, int):
        assert cls_total == cls_base + num_tasks*cls_per_task
    elif isinstance(cls_per_task, list):
        total = 0
        for i in range(cls_per_task):
            total += len(cls_per_task[i])
        assert cls_total == len(cls_base) + int(total)
    else:
        raise TypeError('cls_per_task must be an instance of int or list')
    
    data_idx_per_cls = split_dataset_by_cls(DATASET, num_cls=cls_total)
    
    task_cls_idx = generate_task_by_idx(cls_total, cls_base, num_tasks, cls_per_task, as_list=False)
    if debug is True:
        print(task_cls_idx)
    
    # list of [base_subset, ith_task_subset]
    ret = []
    
    total_idx_list = dict()
    
    # create base task subset
    base_idx_list = task_cls_idx['base']
    base_idx = []
    for i in range(cls_base):
        idx_of_cls = base_idx_list[i]
        all_idx_of_cls = data_idx_per_cls[idx_of_cls]
        base_idx.extend(all_idx_of_cls)
    base_subset = Subset(DATASET, indices=base_idx)
    ret.append(base_subset)
    total_idx_list['base'] = base_idx
    
    #create subsets for tasks
    for i in range(num_tasks):
        ith_task = task_cls_idx[f'task{i}']
        ith_task_idx = []
        for j in range(len(ith_task)):
            idx_of_cls = ith_task[j]
            all_idx_of_cls = data_idx_per_cls[idx_of_cls]
            ith_task_idx.extend(all_idx_of_cls)
        ith_subset = Subset(DATASET, indices=ith_task_idx)
        ret.append(ith_subset)
        total_idx_list[f'task{i}'] = ith_task_idx
        
    task_cls_idx['idx'] = total_idx_list
    
    return ret, task_cls_idx