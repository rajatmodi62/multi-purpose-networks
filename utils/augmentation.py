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
                            A.Normalize(mean= (0.485, 0.456, 0.406),\
                                        std= (0.229, 0.224, 0.225),\
                                        p=1)
                                 ], p=1)

    if mode=='test':
        augmentation= A.Compose([
                            #add randomcrop
                            #add random horizontal flip 
                           
                            A.Normalize(mean= (0.485, 0.456, 0.406),\
                                        std= (0.229, 0.224, 0.225),\
                                        p=1)
                                 ], p=1)
        
    return augmentation


def get_fashion_mnist_augmentation(mode="train"):
    if mode=='train':
        augmentation= A.Compose([
                            #add randomcrop
                            #add random horizontal flip 
                            A.Normalize(mean= (0.485, 0.456, 0.406),\
                                        std= (0.229, 0.224, 0.225),\
                                        p=1)
                                 ], p=1)

    if mode=='test':
        augmentation= A.Compose([
                            #add randomcrop
                            #add random horizontal flip 
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