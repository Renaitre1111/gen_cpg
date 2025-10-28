#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
import numpy as np

import torch
import torchvision
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from tqdm import tqdm
from dataset import SimSiamDataset
import loader
import builder

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch cifar10 finetuning')
parser.add_argument('--data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=800, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch_size', default=512, type=int,
                    metavar='N',
                    help='mini-batch size (default: 512)')
parser.add_argument('--lr', '--learning-rate', default=0.05, type=float,
                    metavar='LR', help='initial (base) learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--save_freq', default=20, type=int, metavar='N',
                    help='save checkpoint every N epochs (default: 20)')
parser.add_argument('--save_dir', default='simsiam/saved_model', help='path to save model')

# simsiam specific configs:
parser.add_argument('--dim', default=2048, type=int,
                    help='feature dimension (default: 2048)')
parser.add_argument('--pred_dim', default=512, type=int,
                    help='hidden dimension of the predictor (default: 512)')
parser.add_argument('--fix_pred_lr', action='store_true',
                    help='Fix learning rate for the predictor')


def main():
    args = parser.parse_args()

    if args.seed is not None:
        set_seed(args.seed)

    save_dir = args.save_dir
    if os.path.exists(save_dir):
        os.makedirs(save_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = builder.SimSiam(
        models.__dict__[args.arch],
        args.dim, args.pred_dim
    )

    model.to(device)
    init_lr = args.lr
    criterion = nn.CosineSimilarity(dim=1).to(device)

    if args.fix_pred_lr:
        optim_params = [{'params': model.encoder.parameters(), 'fix_lr': False},
                        {'params': model.predictor.parameters(), 'fix_lr': True}]
    else:
        optim_params = model.parameters()

    optimizer = torch.optim.SGD(optim_params, init_lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume, map_location=device)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
    
    cudnn.benchmark = False
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    augmentation = [
        transforms.RandomResizedCrop(32, scale=(0.2, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        # transforms.RandomApply([loader.GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]

    lb_idx = np.load("simsiam/label_idx/cifar10/lb_labels_400_100_4600_100_exp_random_noise_0.0_seed_1_idx/lb_labels_400_100_4600_100_exp_random_noise_0.0_seed_1_idx.npy")
    ulb_idx = np.load("simsiam/label_idx/cifar10/lb_labels_400_100_4600_100_exp_random_noise_0.0_seed_1_idx/ulb_labels_400_100_4600_100_exp_random_noise_0.0_seed_1_idx.npy")
    train_idx = np.concatenate((lb_idx, ulb_idx), axis=0)

    all_train_data = torchvision.datasets.CIFAR10(root=args.data, train=True, download=False, transform=None)

    train_data = all_train_data.data[train_idx]
    train_dataset = SimSiamDataset(train_data, loader.TwoCropsTransform(transforms.Compose(augmentation)))
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, init_lr, epoch, args)

        avg_loss = train(train_loader, model, criterion, optimizer, epoch, args, device)
        print(f"==> Epoch [{epoch+1}/{args.epochs}] Completed. \t Average Loss: {avg_loss:.6f}")

        if (epoch + 1) % args.save_freq == 0:
            print(f"==> Saving checkpoint for epoch {epoch + 1}")
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, is_best=False, filename=save_dir + 'checkpoint_{:04d}.pth.tar'.format(epoch+1))

    print("==> Saving final model")
    save_checkpoint({
        'epoch': args.epochs,
        'arch': args.arch,
        'state_dict': model.state_dict(),
        'optimizer' : optimizer.state_dict(),
    }, is_best=True, filename=save_dir + 'checkpoint_final.pth')

def train(train_loader, model, criterion, optimizer, epoch, args, device):
    model.train()
    total_loss = 0.0
    total_batches = 0

    loop = tqdm(train_loader, desc=f"Epoch [{epoch}/{args.epochs}]", leave=False)

    for i, (img1, img2) in enumerate(loop):
        img1 = img1.to(device, non_blocking=True)
        img2 = img2.to(device, non_blocking=True)

        p1, p2, z1, z2 = model(x1=img1, x2=img2)
        loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_postfix(loss=loss.item())

        total_loss += loss.item()
        total_batches += 1

    loop.close()

    return total_loss / total_batches

def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth')

def adjust_learning_rate(optimizer, init_lr, epoch, args):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = cur_lr

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    cudnn.deterministic = True
    warnings.warn('You have chosen to seed training. '
                  'This will turn on the CUDNN deterministic setting, '
                  'which can slow down your training considerably!')
    

if __name__ == '__main__':
    main()



