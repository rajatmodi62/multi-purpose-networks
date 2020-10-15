import os
import numpy as np
import matplotlib.pyplot as plt
import glob as glob
from PIL import Image
import torch
import torch.utils.data as data
import pickle
from pathlib import Path
import cv2
#random sampling in the list 
import random 
#one to one blending for images

class BlendData(data.Dataset):
    #fashion mnist blending factor is to be calculated automatically 

    def __init__(self,cifar_blending_factor=0.5,mode='test'):
        
        self.cifar_blending_factor= cifar_blending_factor
        
        local_data_path = Path('.').absolute()
        cifar_root= local_data_path/'dataset'/'png'/'cifar10'/str(mode)
        cifar_dir= sorted(os.listdir(str(cifar_root)))

        fashion_root= local_data_path/'dataset'/'png'/'fashion_mnist'/str(mode)
        fashion_dir= sorted(os.listdir(str(fashion_root)))

        self.cifar_img_paths=[]
        self.fashion_img_paths=[]
        self.labels= []
        #traverse in the dir 
        for i in range(len(cifar_dir)):
            #create a temp list of both images 
            cifar_temp=[str(cifar_root/cifar_dir[i]/j) for j in os.listdir(str(cifar_root/cifar_dir[i]))]
            fashion_temp=[str(fashion_root/fashion_dir[i]/j) for j in os.listdir(str(fashion_root/fashion_dir[i]))]
            slice= min(len(cifar_temp),len(fashion_temp))
            cifar_temp=random.sample(cifar_temp,slice)
            fashion_temp= random.sample(fashion_temp,slice)
            #append for classes 
            label_temp= [i]*slice
            self.cifar_img_paths+=cifar_temp
            self.fashion_img_paths+=fashion_temp
            self.labels+=label_temp
        #print(len(cifar_img_paths),len(fashion_img_paths),len(labels))

    def __getitem__(self, index):

        #get the item 
        cifar_img= cv2.imread(self.cifar_img_paths[index])
        fashion_img= cv2.imread(self.fashion_img_paths[index])
        #resize 
        fashion_img= cv2.resize(fashion_img,(32,32))
        img=  cv2.addWeighted(cifar_img,self.cifar_blending_factor,fashion_img,1-self.cifar_blending_factor,0)
        label= self.labels[index]
        #meta 
        metadata = {
            'blending_factor':self.cifar_blending_factor
        }
        return img,label,metadata
    
    def __len__(self):
        return len(self.labels)

if __name__ == '__main__':
    dataset= BlendData(cifar_blending_factor=1)
    for i in range(len(dataset)):
        img,label,meta=dataset[i]
        plt.imshow(img)
        plt.show()
        