#Training without embedding layer in the model

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
#argparser arguments 
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')

parser.add_argument("--batch-size", type=int, default=128,
                        help="Training Batch size")
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument("--checkpoint_path", type=str, default="",
                    help="Checkpoint path to load model checkpoint")
parser.add_argument("--training_type", type=str, default="cifar",
                    help="type of training (cifar/fashion_mnist/combined")
parser.add_argument("--num-workers", type=int, default=2,
                        help="Number of workers for dataloaders")
# parser.add_argument("--embedding_layer",action='store_true',
#                     help="Switch this flag on if embedding layer is to be trained")
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch


#returns trainloader and testloader 
def get_dataloaders():

    if args.training_type=='cifar':
        trainset= CIFAR10(data_root="dataset/cifar10",
                      transform=None,
                      mode='train')
        testset= CIFAR10(data_root="dataset/cifar10",
                      transform=None,
                      mode='test')
    elif args.training_type== 'fashion_mnist':
        trainset= FashionMNIST(data_root="dataset/fashion-mnist",
                           transform=None,
                           mode='train')
        testset= FashionMNIST(data_root="dataset/fashion-mnist",
                           transform=None,
                           mode='test')
    elif args.training_type== 'combined':
        trainset= CIFAR_FASHMNIST(cifar_data_root="dataset/cifar10",\
                            fashion_mnist_data_root="dataset/fashion-mnist",\
                            transform=None,\
                            mode='train')
        testset= CIFAR_FASHMNIST(cifar_data_root="dataset/cifar10",\
                            fashion_mnist_data_root="dataset/fashion-mnist",\
                            transform=None,\
                            mode='test')
    #create dataloaders 
    trainloader = torch.utils.data.DataLoader(
                    trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    testloader = torch.utils.data.DataLoader(
                    testset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
       
    return trainloader,testloader

def train_single_dataset(epoch):
    
    print('\nEpoch: %d' % epoch)
    model.train()
    classifier.train()
    train_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets,_) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        inputs = inputs.permute(0,3,1,2)
        # print("inputs shape",inputs.shape)
        optim_model.zero_grad()
        optim_classifier.zero_grad()
        outputs = model(inputs)
        outputs = classifier(outputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optim_classifier.step()
        optim_model.step()
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        # print("predicted",predicted)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        
        
def test_single_dataset(epoch):
    print("in testing code")
    global best_acc
    model.eval()
    classifier.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets,_) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = inputs.permute(0,3,1,2)
            outputs = model(inputs)
            outputs = classifier(outputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
            

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'model': model.state_dict(),
            'classifier': classifier.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        # if not os.path.isdir('checkpoint'):
        #     os.mkdir('checkpoint')
        #dump the dictionary to the 
        torch.save(state, str(save_dir/'checkpoint.pth'))
        best_acc = acc


###################################### TRAINING STARTS HERE ############################
local_data_path = Path('.').absolute()
#create experiment 
experiment= args.training_type
save_dir= (local_data_path/'experiments'/experiment)
(save_dir).mkdir(exist_ok=True, parents=True)

#get dataloaders
trainloader,testloader= get_dataloaders()
#get model without embedding 
model= Backbone()
#get classifier 
if args.training_type=="combined":
    classifier= ClassificationHead(num_classes=20)
else:
    classifier= ClassificationHead(num_classes=10)

#create loss 
criterion= nn.CrossEntropyLoss()

#create optimizer
optim_model= optim.SGD(model.parameters(), lr=args.lr,
                    momentum=0.9, weight_decay=5e-4)
optim_classifier= optim.SGD(classifier.parameters(), lr=args.lr,
                    momentum=0.9, weight_decay=5e-4)

############ CODE FOR RESUMING THE TRAINING ###########################################
if args.checkpoint_path!="":
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    classifier.load_state_dict(checkpoint['classifier'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']   


def update_learning_rate(epoch):
    #update model lr
    if epoch<150:
        learning_rate= 0.1
    elif 150<=epoch<250:
        learning_rate= 0.01
    else:
        learning_rate = 0.001
    #update learning rate
    for param_group in optim_model.param_groups:
        param_group['lr'] = learning_rate
    # update classifier learning rate
    for param_group in optim_classifier.param_groups:
        param_group['lr'] = learning_rate


def main():

    #apply the training schedue 
    for epoch in range(start_epoch, start_epoch+350):
        #call train
        update_learning_rate(epoch)
        train_single_dataset(epoch)
        test_single_dataset(epoch)

        print("epoch: ",epoch,"best accuracy found is: ",best_acc)


    print("overall best accuracy found is: ",best_acc)
if __name__ == '__main__':
    main()