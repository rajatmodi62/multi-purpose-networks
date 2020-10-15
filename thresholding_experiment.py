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

#drw tsne
from tsne import draw_tsne

#argparser
parser = argparse.ArgumentParser(description='Alternate conditioning testing for CIFAR/FashionMNIST')


parser.add_argument("--cifar_checkpoint_path", type=str, default="experiments/conditioned/cifar/checkpoint.pth",
                    help="Enter the checkpoint for  CIFAR")
parser.add_argument("--fashion_mnist_checkpoint_path", type=str, default="experiments/conditioned/fashion_mnist/checkpoint.pth",
                    help="Enter the checkpoint for fashion_mnist CIFAR")
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
classifier_cifar= ClassificationHead(num_classes=10).to(device)
classifier_fashion_mnist= ClassificationHead(num_classes=10).to(device)
#load the checkpoint
cifar_checkpoint= torch.load(args.cifar_checkpoint_path)
fashion_mnist_checkpoint= torch.load(args.fashion_mnist_checkpoint_path)
#for checking on cifar dataset
model.load_state_dict(cifar_checkpoint['model'])
#for checking on fashion mnist dataset
#model.load_state_dict(fashion_mnist_checkpoint['model'])
#load appropriate classifier weights and get testset
dataset_name= "cifar"
threshold_confidence= 0.8
embedding_label= 1
classifier_cifar.load_state_dict(cifar_checkpoint['classifier_cifar'])
classifier_fashion_mnist.load_state_dict(cifar_checkpoint['classifier_fashion_mnist'])

testset = CIFAR10(data_root="dataset/cifar10",
                    transform=None,
                    mode='test')
# else:
#testset = FashionMNIST(data_root="dataset/fashion-mnist",
#                                        transform=None,
#                                        mode='test')
testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

#simple testing code
def test():
    print("in testing code")
    global best_acc
    model.eval()
    classifier_cifar.eval()
    classifier_fashion_mnist.eval()
    test_loss = 0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        #cifar fwd pass
        correct=0

        for batch_idx, (inputs, targets, meta) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = inputs.permute(0, 3, 1, 2)
            #convert embedding_label to tensor
            # print("inputs shape",inputs.shape)
            n_repeats= inputs.shape[0]
            embedding_labels = torch.ones(1)*embedding_label
            embedding_labels= embedding_labels.type(torch.LongTensor).to(device)
            embedding_labels = torch.cat(n_repeats*[embedding_labels])
            #print("shape of embedding labels",embedding_labels.shape)
            embedding_labels = embedding_labels.unsqueeze(1)
            outputs = model(inputs, embedding_labels)
            outputs = classifier_cifar(outputs)
            #print("outputs shape",outputs.shape)
            #print(outputs)
            loss = criterion(outputs, targets)
            #print("probabilities",probabilities)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            #get probabilities 
            probabilities= F.softmax(outputs)
            #get probabilities for predicted classes 
            probabilities= torch.gather(probabilities,1, predicted.unsqueeze(1))
            #print("max probabilities",probabilities[:,predicted])
            #print("predicted size",predicted.size())
            #print("probabilities size",probabilities.size())
            
            total += targets.size(0)
            
            #correct += predicted.eq(targets).sum().item()
            for idx,is_correct in enumerate(predicted.eq(targets)):
               if is_correct==True:
                #check if the probability at that index is greater than threshold
                #print("nips",probabilities[idx][0],threshold_confidence)
                if probabilities[idx][0]>=threshold_confidence:
                    correct+=1
                    #print("true")
              
                #print("rajat",probabilities.size())
               
            #print("before indexing",probabilities[0:10])
            #print("mask",predicted.eq(targets)[0:10])
            #print("after indexing",probabilities[predicted.eq(targets)][0:10])
            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    cifar_acc = 100.*correct/total
    #print("Accuracy is",acc, "for dataset",args.dataset, " conditioning label ", embedding_label )

    test_loss = 0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    correct=0

    with torch.no_grad():
        #fashion_mnist fwd pass
        for batch_idx, (inputs, targets, meta) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = inputs.permute(0, 3, 1, 2)
            #convert embedding_label to tensor
            # print("inputs shape",inputs.shape)
            n_repeats= inputs.shape[0]
            embedding_labels = torch.ones(1)*embedding_label
            embedding_labels= embedding_labels.type(torch.LongTensor).to(device)
            embedding_labels = torch.cat(n_repeats*[embedding_labels])
            print("shape of embedding labels",embedding_labels.shape)
            embedding_labels = embedding_labels.unsqueeze(1)
            outputs = model(inputs, embedding_labels)
            outputs = classifier_fashion_mnist(outputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            #get probabilities 
            probabilities= F.softmax(outputs)
            #get probabilities for predicted classes 
            probabilities= torch.gather(probabilities,1, predicted.unsqueeze(1))
            #print("max probabilities",probabilities[:,predicted])
            #print("predicted size",predicted.size())
            #print("probabilities size",probabilities.size())
            
            total += targets.size(0)
            
            #correct += predicted.eq(targets).sum().item()
            for idx,is_correct in enumerate(predicted.eq(targets)):
               if is_correct==True:
                #check if the probability at that index is greater than threshold
                #print("nips",probabilities[idx][0],threshold_confidence)
                if probabilities[idx][0]>=threshold_confidence:
                    correct+=1
                    #print("true")
                
                #print(#"rajat",probabilities.size())
               
            #print("before indexing",probabilities[0:10])
            #print("mask",predicted.eq(targets)[0:10])
            #print("after indexing",probabilities[predicted.eq(targets)][0:10])
            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    fashion_mnist_acc = 100.*correct/total

    print("accuracy of cifar",cifar_acc, "accuracy of fashion mnist", fashion_mnist_acc, "dataset", args.dataset, "conditioning label", embedding_labels)
#calling test code
test()
local_data_path = Path('.').absolute()
(local_data_path/'visualization').mkdir(exist_ok=True, parents=True)
save_path= str(local_data_path/'visualization')+'/'+dataset_name+'.png'#
#draw_tsne(model,testloader,embedding_label,save_path)

