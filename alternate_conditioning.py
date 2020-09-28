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

#import progressbar
from utils.utils import progress_bar


#argparser
parser = argparse.ArgumentParser(description='Alternate conditioning testing for CIFAR/FashionMNIST')

parser.add_argument("--checkpoint_path", type=str, default="",
                    help="Enter the checkpoint for either CIFAR/FashionMNIST")
parser.add_argument("--dataset", type=str, default="CIFAR",
                    help="CIFAR10's checkpoint")

#Some redundant options for making testloaders 
parser.add_argument("--batch-size", type=int, default=128,
                    help="Training Batch size")
parser.add_argument("--num-workers", type=int, default=0,
                    help="Number of workers for dataloaders")

                   
args = parser.parse_args()

#define device 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#intialize the model 
model = Backbone(apply_embedding=True).to(device)

#get a classifier [name is independendent of cifar/fashion_mnist now]
classifier= ClassificationHead(num_classes=10).to(device)

#load the checkpoint 
checkpoint= torch.load(args.checkpoint_path)
model.load_state_dict(checkpoint['model'])

#load appropriate classifier weights and get testset

if args.dataset=='CIFAR':
    classifier.load_state_dict(checkpoint['classifier_cifar'])
    testset = CIFAR10(data_root="dataset/cifar10",
                            transform=None,
                            mode='test')
    # Note: I am intentionally keeping wrong label, because we want to check if external conditioning has impact on results
    embedding_label= 1
else:
    classifier.load_state_dict(checkpoint['classifier_fashion_mnist'])
    testset = FashionMNIST(data_root="dataset/fashion-mnist",
                                        transform=None,
                                        mode='test')
    # Note: I am intentionally keeping wrong label, because we want to check if external conditioning has impact on results
    embedding_label = 0

testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

#simple testing code 
def test():
    print("in testing code")
    global best_acc
    model.eval()
    classifier.eval()
    test_loss = 0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch_idx, (inputs, targets, meta) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = inputs.permute(0, 3, 1, 2)
            #convert embedding_label to tensor
            # print("inputs shape",inputs.shape)
            n_repeats= inputs.shape[0]
            embedding_labels= torch.Tensor(embedding_label).to(device)
            embedding_labels= embedding_labels.type(torch.LongTensor).to(device)
            embedding_labels = torch.cat(n_repeats*[embedding_labels])
            embedding_labels = embedding_labels.unsqueeze(1)
            outputs = model(inputs, embedding_labels)
            outputs = classifier(outputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    acc = 100.*correct/total
    print("Accuracy is",acc, "for dataset",args.dataset, " conditioning label ", embedding_label )
#calling test code 
test()