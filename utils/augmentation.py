import numpy as np
import math
import cv2
from PIL import Image
import albumentations as A

def get_cifar_augmentation(mode="train"):
    # print("cifar augmentation called")
    if mode=='train':
        augmentation= A.Compose([
                            #add randomcrop
                            #add random horizontal flip
                            A.PadIfNeeded((32+8),(32+8)),\
                            A.RandomCrop(32,32,p=1),\
                            A.HorizontalFlip(p=0.5),\
                            A.Normalize(mean= (0.4914, 0.4822, 0.4465),\
                                        std= (0.2023, 0.1994, 0.2010),\
                                        p=1)
                                 ], p=1)

    if mode=='test':
        augmentation= A.Compose([
                            
                            A.Normalize(mean= (0.4914, 0.4822, 0.4465),\
                                        std= (0.2023, 0.1994, 0.2010),\
                                        p=1)
                                 ], p=1)
        
    return augmentation


def get_fashion_mnist_augmentation(mode="train"):
    if mode=='train':
        augmentation= A.Compose([
                            #add randomcrop
                            #add random horizontal flip 
                            A.HorizontalFlip(p=0.5),\
                            A.Resize(32,32,p=1),
                            A.Normalize(mean= (0.485, 0.456, 0.406),\
                                        std= (0.229, 0.224, 0.225),\
                                        p=1)
                            
                                 ], p=1)

    if mode=='test':
        augmentation= A.Compose([
                            #add randomcrop
                            #add random horizontal flip
                            A.Resize(32,32,p=1), 
                            A.Normalize(mean= (0.485, 0.456, 0.406),\
                                        std= (0.229, 0.224, 0.225),\
                                        p=1)
                                 ], p=1)
        
    return augmentation

def get_cifar_fashmnist_augmentation(mode="train"):
    if mode=='train':
        augmentation= A.Compose([
                            #add randomcrop
                            #add random horizontal flip 
                            A.HorizontalFlip(p=0.5),\
                            A.Normalize(mean= (0.485, 0.456, 0.406),\
                                        std= (0.229, 0.224, 0.225),\
                                        p=1),\
                            A.Resize(28,28,p=1)
                                 ], p=1)
    if mode=='test':
        augmentation= A.Compose([
                            #add randomcrop
                            #add random horizontal flip 
                            A.Normalize(mean= (0.485, 0.456, 0.406),\
                                        std= (0.229, 0.224, 0.225),\
                                        p=1),\
                            A.Resize(28,28,p=1)
                                 ], p=1)
        
    return augmentation