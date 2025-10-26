# code reference: https://colab.research.google.com/drive/1IJkrrV-D7boSCLVKhi7t5docRYqORtm3#scrollTo=jpy3GC7XzC7J
import os
import argparse

import time
import logging
import random
import math
from copy import deepcopy
from contextlib import contextmanager
from collections import defaultdict

import torch
from torch import optim, nn
from torch.nn import functional as F
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize
from torch.utils.data import TensorDataset, DataLoader

def label_to_noise(labels, class_num, channel_num=1, sequence_length=128, variance=0.001, seed=0):
    gen = torch.Generator()
    gen.manual_seed(seed)

    class_means = torch.randn(class_num, channel_num, sequence_length, generator=gen)
    means = class_means.index_select(0, labels)

    std = math.sqrt(variance)
    noise = torch.randn_like(means, generator=gen).mul_(std).add_(means)

    return noise