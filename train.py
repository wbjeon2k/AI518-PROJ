# dependencies
import torch
import torch.nn as nn
import torch.utils
from torch.utils.data import Dataset, DataLoader
import torch.utils.data
import torchvision.transforms as TF
import yaml
from tqdm import tqdm
import hashlib
import time
import json


# import models
from models import GAN, Glow, iGPT, VAE

# custom utils
from data import classincremental
from utils import config

# implement training loop for the generated CIL task

def test_performance_of_task(MODEL : nn.Module, task_set : Dataset):
    MODEL.eval()
    TEST_LOADER = DataLoader(
        task_set, batch_size=256, num_workers=4
    )
    with torch.no_grad():
        test_loss = 0
        for x,_ in TEST_LOADER:
            batch_loss = MODEL.testing(x)
            test_loss += batch_loss.item()
            
        return test_loss

def train_CIL(MODEL: nn.Module):
    # get config from config.yaml
    cfg_dict = config.load_config_from_yaml()
    
    #get train configs
    dset_name, dset_path = config.get_train_set_config(cfg_dict)
    
    #train for the base task
    TRAIN_TRANSFORM = config.get_transform(is_train=True)
    TEST_TRANSFORM = config.get_transform(is_train=False)
    TRAIN_SET_WHOLE = config.get_dataset(dset_name, dset_path, TRAIN_TRANSFORM,is_train=True)
    TEST_SET_WHOLE = config.get_dataset(dset_name, dset_path, TEST_TRANSFORM, is_train=False)
    
    cls_total, cls_base, num_tasks, cls_per_task = config.get_task_info(cfg_dict)
    
    assert config.get_total_number_of_cls(dset_name) == cls_total
    
    # task_to_train : [train_set_base, train_set_task0, ....]
    task_to_train, train_set_info = classincremental.generate_CIL_task(
        DATASET=TRAIN_SET_WHOLE, cls_total = cls_total, 
        cls_base = cls_base, num_tasks = num_tasks, cls_per_task = cls_per_task
    )
    train_set_info['dataset'] = dset_name
    
    # task_to_test : [test_set_base, test_set_task0, ....]
    task_to_test, test_set_info = classincremental.generate_CIL_task(
        DATASET=TEST_SET_WHOLE, cls_total = cls_total, 
        cls_base = cls_base, num_tasks = num_tasks, cls_per_task = cls_per_task
    )
    test_set_info['dataset'] = dset_name
    
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
    
    def train_one_epoch(MODEL: nn.Module, DATALOADER):
        """
        TODO:
        implement a generic training loop looks like below.
        Dataloader =>>>>
        """
        for x, _ in DATALOADER:
            MODEL.learning(x)

    for i in tqdm(range(epoch_base)):
        train_one_epoch(MODEL, BASE_LOADER)
    
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
            train_one_epoch(MODEL,ITH_TASK_LOADER)
        
        #measure performance for the base task
        #after training each task    
        TRAIN_RESULT[f'task{i}'] = test_performance_of_task(MODEL, BASE_TEST_SET)
        
    # save training results
    hashlib.sha1().update(str(time.time()).encode("utf-8"))
    hashlib.md5().update(str(time.time()).encode("utf-8"))
    experiment_key = str(hashlib.sha1().hexdigest()[30:40]) + \
        str(hashlib.md5().hexdigest()[0:5])
    
    with open(f'result/{experiment_key}_train_set_info.json', 'w') as outfile:
        json.dump(train_set_info, outfile, ensure_ascii=True, indent=4)
        
    with open(f'result/{experiment_key}_test_set_info.json', 'w') as outfile:
        json.dump(test_set_info, outfile,ensure_ascii=True, indent=4)
        
    with open(f'result/{experiment_key}_train_result.json', 'w') as f:
        json.dump(TRAIN_RESULT, f, ensure_ascii=True, indent=4)
    
    cfg_dict['model_type'] = MODEL.__class__.__name__
    cfg_dict['experiment_key'] = experiment_key
    with open(f'result/{experiment_key}_cfg.json', 'w') as f:
        json.dump(cfg_dict, f, ensure_ascii=True, indent=4)
        
    torch.save(MODEL.state_dict(), f'result/{experiment_key}_model.pth')
    # end of CIL training
    
def get_test_metric(metric_name):
    """"""

import torchvision.transforms as TF
def generate_task_to_test(experiment_key, DATASET: Dataset):
    """
    read test_set_info.json, reconstruct test set for base task
    """
    with open(f'result/{experiment_key}_test_set_info.json', 'r') as f:
        test_info_dict = json.load(f)
    
    base_idx = test_info_dict['idx']['base']
    base_subset = torch.utils.data.Subset(DATASET, base_idx)
    return base_subset
    
def get_samples_from_model(MODEL: nn.Module):
    """
    get samples from model by num_samples
    """
    MODEL.eval()
    ret = MODEL.sample()
    if isinstance(ret, torch.Tensor):
        ret = ret.detach().cpu().numpy()
    return ret
    
def get_model_pth_from_key(MODEL : nn.Module, experiment_key):
    """
    load model from /result/{experiment_key}_model.pth
    """
    MODEL.load_state_dict(torch.load(f'./result/{experiment_key}_model.pth'))
    return MODEL
    
# def test_performance_of_task(MODEL : nn.Module, task_set : Dataset):
#     TEST_LOADER = DataLoader(
#         task_set, batch_size=256, num_workers=4
#     )
#     test_loss = 0
#     for x,_ in TEST_LOADER:
#         batch_loss = MODEL.testing(x)
#         test_loss += batch_loss.item()
    
#     return test_loss

import os
from PIL import Image
import numpy as np

def test_CIL(MODEL: nn.Module):
    cfg_load = config.load_config_from_yaml(is_train=False)
    cfg_dict = cfg_load['test']
    
    experiment_key = cfg_dict['experiment_key']
    #metric_name = cfg_dict['metric_name']
    #num_samples = cfg_dict['num_samples']
    model_args = cfg_dict['model_args']
    dset_name = cfg_dict['dataset_name']
    dset_path = cfg_dict['dataset_path']
    
    transform = config.get_transform(is_train=False)
    #TEST_METRIC = get_test_metric(metric_name)
    WHOLE_TEST_SET = config.get_dataset(dset_name, dset_path, transform)
    TASK_TO_TEST = generate_task_to_test(experiment_key, WHOLE_TEST_SET)
    MODEL = get_model_pth_from_key(MODEL, experiment_key)
    MODEL_TYPE = MODEL.__class__.__name__
    
    test_loss = test_performance_of_task(MODEL, TASK_TO_TEST)
    
    test_result_dict = dict()
    
    test_result_dict['experiment_key'] = experiment_key
    test_result_dict['test_loss'] = test_loss
    
    try:
        os.makedirs(f'./result/test_result/{experiment_key}', exist_ok=False)
    except:
        pass
    
    samples = get_samples_from_model(MODEL)
    print(samples.shape)
    B,H,W,C = samples.shape
    sample_path_list = []
    for i in range(B):
        im = Image.fromarray(samples[i,:,:,:].astype(np.uint8))
        im.save(f'./result/test_result/{experiment_key}/sample_{i}.png', "PNG")
        sample_path_list.append(f'/result/test_result/{experiment_key}/sample_{i}.png')
        
    test_result_dict['samples'] = sample_path_list
    
    with open(f'./result/{experiment_key}_test_result.json', 'w') as outfile:
        json.dump(test_result_dict, outfile,ensure_ascii=True, indent=4)