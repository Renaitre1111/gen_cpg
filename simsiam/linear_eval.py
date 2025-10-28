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
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img = self.data[idx]
        target = self.targets[idx]
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        
        if self.transform is not None:
            img = self.transform(img)
        return img, target
    
def main():
    args = parser.parse_args()

    if args.seed is not None:
        set_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = builder.SimSiam(
        models.__dict__[args.arch],
        args.dim, args.pred_dim
    )
    model.to(device)

    checkpoint = torch.load(args.pretrained, map_location=device)

    model.load_state_dict(checkpoint['state_dict'])

    for param in model.encoder.parameters():
        param.requires_grad = False
    
    in_features = model.encoder.fc[0].in_features
    model.encoder.fc = nn.Linear(in_features, 10)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 80], gamma=0.1)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    val_transforms = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    val_dataset = datasets.CIFAR10(args.data, train=False, transform=val_transforms, download=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    train_transform = torch.Compose([
        transforms.RandomResizedCrop(32, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
    all_train_data = datasets.CIFAR10(args.data, train=True, download=True)
    lb_idx = np.load(args.lb_idx_path)

    train_data = all_train_data.data[lb_idx]
    train_labels = np.array(all_train_data.targets)[lb_idx]
    train_dataset = LinearevalDataset(train_data, train_labels, transform=train_transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                                               num_workers=args.workers, pin_memory=True, drop_last=True)
    
    best_acc = 0.0
    for epoch in range(args.epochs):
        train_loss = train(train_loader, model, criterion, optimizer, epoch, device)

        val_loss, val_acc = validate(val_loader, model, criterion, device)

        scheduler.step()

        if val_acc > best_acc:
            best_acc = val_acc
        
        print(f'Epoch {epoch+1}/{args.epochs} Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f} Best Acc: {best_acc:.4f}')

    print(f"Best accurary: {best_acc:.4f}")

def train(train_loader, model, criterion, optimizer, epoch, device):
    model.train()
    total_loss = 0.0
    total_batches = 0

    loop = tqdm(train_loader, desc=f'Epoch {epoch+1} Train', leave=False)
    for imgs, targets in loop:
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        output = model.encoder(imgs)
        loss = criterion(output, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_batches += 1
        loop.set_postfix(loss=loss.item())

    loop.close()
    return total_loss / total_batches

def validate(val_loader, model, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    loop = tqdm(val_loader, desc=f'Validate', leave=False)
    with torch.no_grad():
        for imgs, targets in loop:
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            output = model.encoder(imgs)
            loss = criterion(output, targets)

            total_loss += loss.item()

            _, predicted = torch.max(output, 1)
            total_samples += targets.size(0)
            total_correct += (predicted == targets).sum().item()

        loop.close()
        avg_loss = total_loss / len(val_loader)
        acc = 100.0 * total_correct / total_samples
        return avg_loss, acc

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    cudnn.deterministic = True
    warnings.warn('You have chosen to seed training. '
                  'This will turn on the CUDNN deterministic setting, '
                  'which can slow down your training considerably!')