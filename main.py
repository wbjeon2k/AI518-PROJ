# dependencies
import data
import models
import torch
import torch.nn as nn
import argparse
from models.GAN import GAN as GAN
from models.Glow import Glow as Glow
from models.VAE import VAE as VAE
from models.iGPT import AutoregressiveTransformer as iGPT
from models.PixelCNN import PixelCNN
import train

exec_mode = ['train', 'test']
model_choices = ['GAN', 'Glow', 'iGPT', 'VAE', 'PixelCNN']

def get_model(MODEL_TYPE):
    MODEL = None
    if MODEL_TYPE == model_choices[0]:
        MODEL = GAN()
    elif MODEL_TYPE == model_choices[1]:
        MODEL = Glow()
    elif MODEL_TYPE == model_choices[2]:
        MODEL = iGPT()
    elif MODEL_TYPE == model_choices[3]:
        MODEL = VAE()
    elif MODEL_TYPE == model_choices[4]:
        MODEL = PixelCNN()
    else:
        raise RuntimeError
    return MODEL

def train_model(MODEL_TYPE):
    """
    train generative models in a class-incremental way
    form of train loop TBD
    """
    MODEL = get_model(MODEL_TYPE)
    train.train_CIL(MODEL)

def test_model(MODEL_TYPE):
    """
    measure the performance of the first task (base task)
    """
    MODEL = get_model(MODEL_TYPE)
    train.test_CIL(MODEL)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='AI518 Project'
    )
    
    parser.add_argument('--mode', dest='mode', choices=exec_mode)
    parser.add_argument('--model', dest='model', choices=model_choices, type=str)
    
    args = parser.parse_args()
    
    if args.mode == exec_mode[0]: #train
        train_model(args.model)
    elif args.mode == exec_mode[1]: #test
        test_model(args.model)
    else:
        raise TypeError