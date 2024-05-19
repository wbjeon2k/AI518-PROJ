# dependencies
import torch
import torch.nn as nn
import yaml

from data import classincremental
from utils import config

# implement training loop for the generated CIL task

def train_CIL(MODEL: nn.Module, config: dict):
    