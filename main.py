# dependencies
import data
import models
import torch
import torch.nn as nn
import argparse

def train_model():
    """
    train generative models in a class-incremental way
    form of train loop TBD
    """


def test_model():
    """
    measure the performance of the first task (base task)
    """
    
exec_mode = ['train', 'test']

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='AI518 Project'
    )
    
    parser.add_argument('--mode', dest='mode', choices=exec_mode)
    
    args = parser.parse_args()
    
    if args.mode == exec_mode[0]: #train
        train_model()
    elif args.mode == exec_mode[1]: #test
        test_model()
    else:
        raise TypeError