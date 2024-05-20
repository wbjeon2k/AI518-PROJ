# dependencies
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import yaml
from tqdm import tqdm
import hashlib
import time
import json

# custom utils
from data import classincremental
from utils import config

# implement training loop for the generated CIL task

def test_performance_of_task(MODEL : nn.Module, task_set : Dataset):
    """
    TODO: make a test loop that takes task_set
    evaluate test loss for the task_set
    """
    pass

def train_CIL(MODEL: nn.Module):
    # get config from config.yaml
    cfg_dict = config.load_config_from_yaml()
    
    #get train configs
    dset_name, dset_path = config.get_train_hyper_parameters(cfg_dict)
    
    #train for the base task
    TRAIN_SET_WHOLE = config.get_dataset(dset_name, dset_path, is_train=True)
    TEST_SET_WHOLE = config.get_dataset(dset_name, dset_path, is_train=False)
    
    cls_total, cls_base, num_tasks, cls_per_task = config.get_task_info(cfg_dict)
    
    assert config.get_total_number_of_cls(dset_name) == cls_total
    
    # task_to_train : [train_set_base, train_set_task0, ....]
    task_to_train, train_set_info = classincremental.generate_CIL_task(
        DATASET=TRAIN_SET_WHOLE, cls_total = cls_total, 
        cls_base = cls_base, num_tasks = num_tasks, cls_per_task = cls_per_task
    )
    
    # task_to_test : [test_set_base, test_set_task0, ....]
    task_to_test, test_set_info = classincremental.generate_CIL_task(
        DATASET=TEST_SET_WHOLE, cls_total = cls_total, 
        cls_base = cls_base, num_tasks = num_tasks, cls_per_task = cls_per_task
    )
    
    optim_name, learning_rate, epoch_base, \
        epoch_per_task, batch_size = config.get_train_hyper_parameters(cfg_dict)
    
    # train to base task
    BASE_TRAIN_SET = task_to_train[0]
    BASE_LOADER = DataLoader(
        BASE_TRAIN_SET, batch_size=batch_size,
        shuffle=True, num_workers=4, drop_last=True
    )
    OPTIM = config.get_optim(optim_name, MODEL, lr=learning_rate)
    
    TRAIN_RESULT = dict()
    
    def train_one_epoch(MODEL: nn.Module, DATALOADER, LOSSFUNC, OPTIM):
        """
        TODO:
        implement a generic training loop looks like below.
        """
        # for x,_ in BASE_LOADER:
        #     pred = MODEL(x)
        #     loss = ...
        #     OPTIM.zero_grad()
        #     loss.backward()
        #     OPTIM.step()
    
    for i in tqdm(range(epoch_base)):
        train_one_epoch(MODEL)
    
    BASE_TEST_SET = task_to_test[0]
    TRAIN_RESULT['base'] = test_performance_of_task(MODEL, BASE_TEST_SET)
    
    for i in tqdm(range(num_tasks)):
        ITH_TASK_SET = task_to_train[i+1] #task_to_train[0] is base_task
        ITH_TASK_LOADER = DataLoader(
            ITH_TASK_SET, batch_size=batch_size,
            shuffle=True, num_workers=4, drop_last=True
        )
        for j in range(epoch_per_task):
            """
            TODO: replace with proper train_one_epoch
            """
            # train_one_epoch(MODEL,ITH_TASK_LOADER, )
        
        #measure performance for the base task
        #after training each task    
        TRAIN_RESULT[f'task{i}'] = test_performance_of_task(MODEL, BASE_TEST_SET)
        
    # save training results
    hashlib.sha1().update(str(time.time()).encode("utf-8"))
    experiment_key = str(hashlib.sha1().hexdigest()[:10])
    
    with open(f'{experiment_key}_train_set_info.yml', 'w') as outfile:
        yaml.dump(train_set_info, outfile, default_flow_style=True)
        
    with open(f'{experiment_key}_test_set_info.yml', 'w') as outfile:
        yaml.dump(test_set_info, outfile, default_flow_style=True)
        
    with open(f'{experiment_key}_train_result.json', 'w') as f:
        json.dump(TRAIN_RESULT, f, ensure_ascii=True, indent=4)
        
    torch.save(MODEL.state_dict(), f'{experiment_key}_model.pth')
    # end of CIL training