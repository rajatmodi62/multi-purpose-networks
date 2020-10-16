# testing without embedding layer 
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import os
import argparse

from pathlib import Path
import tqdm

#import models
from models.backbone import Backbone
from models.classification_head import ClassificationHead

#import dataloaders
from dataloader.cifar10 import CIFAR10
from dataloader.fashion_mnist import FashionMNIST
from dataloader.cifar_fashmnist import CIFAR_FASHMNIST

#import progressbar
from utils.utils import progress_bar

from utils.variables import classifier_dict
from tsne import draw_tsne_combined

# argparser arguments
parser = argparse.ArgumentParser(description='Combined testing module')

parser.add_argument("--batch-size", type=int, default=128,
                    help="Training Batch size")
parser.add_argument("--checkpoint_path", type=str, default="",
                    help="Checkpoint path to load model checkpoint")
parser.add_argument("--backbone", type=str, default="resnet18",
                    help="BACKBONE TO TRAIN WITH:resnet18/resnet50/resnest50")
parser.add_argument("--num-workers", type=int, default=0,
                    help="Number of workers for dataloaders")
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_dataloaders():
    trainset = CIFAR_FASHMNIST(cifar_data_root="dataset/cifar10",
                                   fashion_mnist_data_root="dataset/fashion-mnist",
                                   transform=None,
                                   mode='train')
    testset_cifar = CIFAR10(data_root="dataset/cifar10",
                            transform=None,
                            mode='test')
    testset_fashion_mnist = FashionMNIST(data_root="dataset/fashion-mnist",
                                            transform=None,
                                            mode='test')
     # create dataloaders
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    testloader_cifar = torch.utils.data.DataLoader(
        testset_cifar, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    testloader_fashion_mnist = torch.utils.data.DataLoader(
        testset_fashion_mnist, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    return trainloader, testloader_cifar, testloader_fashion_mnist


#get dataloader 
trainloader, testloader_cifar, testloader_fashion_mnist = get_dataloaders()

#get model 
model = Backbone(backbone=args.backbone).to(device)
model.load_state_dict(torch.load(args.checkpoint_path)['model'])

#no need of classification head, since we are only drawing tsne 

#cifar testing
draw_tsne_combined(model,testloader_cifar,save_path='visualization/tsne.png')