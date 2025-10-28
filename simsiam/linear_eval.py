#!/usr/bin/env python
import argparse
import os
import random
import warnings
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm
import builder

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 Linear Evaluation')
parser.add_argument('--data', metavar='DIR', required=True,
                    help='path to dataset')
parser.add_argument('--pretrained', type=str, required=True,
                    help='path to your pretrained simsiam checkpoint (e.g., checkpoint_final.pth)')
parser.add_argument('--lb_idx_path', type=str, 
                    default='simsiam/label_idx/cifar10/lb_labels_400_100_4600_100_exp_random_noise_0.0_seed_1_idx/lb_labels_400_100_4600_100_exp_random_noise_0.0_seed_1_idx.npy',
                    help='path to the numpy file of labeled indices')
parser.add_argument('--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names))
parser.add_argument('--workers', default=16, type=int,
                    help='number of data loading workers')
parser.add_argument('--epochs', default=100, type=int,
                    help='number of total epochs to run')
parser.add_argument('--batch_size', default=256, type=int,
                    help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=0, type=float,
                    metavar='W', help='weight decay (default: 0)',
                    dest='weight_decay')
parser.add_argument('--seed', default=1, type=int,
                    help='seed for initializing training.')
parser.add_argument('--dim', default=2048, type=int,
                    help='feature dimension (default: 2048)')
parser.add_argument('--pred_dim', default=512, type=int,
                    help='hidden dimension of the predictor (default: 512)')

class LinearevalDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets