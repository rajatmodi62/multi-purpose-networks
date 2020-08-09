import os
import numpy as np
import matplotlib.pyplot as plt
import glob as glob
from PIL import Image
import torch
import torch.utils.data as data
import pickle
from pathlib import Path
from utils.augmentation import get_fashion_mnist_augmentation
# Classes for this dataset are:
['T-shirt/top',
 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


class FashionMNIST(data.Dataset):

    def __init__(self, data_root, transform=None, mode='train'):
        super().__init__()
        print("fashion mnist called")
        if transform is not None:
            self.transform = transform
        else:
            print("fetching default transform")
            self.transform = get_fashion_mnist_augmentation(mode=mode)
        self.mode = mode
        self.local_data_path = Path('.').absolute()
        self.data_root = data_root

        # dump the training file in the disk
        if mode == 'train':
            self.data_file = 'processed/training.pt'
        else:
            self.data_file = 'processed/test.pt'

        # read the data
        self.data, self.targets = torch.load(
            str(self.local_data_path/self.data_root/self.data_file))
        self.classes = [
            'T-shirt/top',
            'Trouser',
            'Pullover',
            'Dress',
            'Coat',
            'Sandal',
            'Shirt',
            'Sneaker',
            'Bag',
            'Ankle boot'
        ]

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        # unsqueeze image
        img = img.numpy()
        # repeat image along three axis for compatible cnn input
        img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
        # print("type of img",img.dtype,type(img),img.shape)
        img = self.transform(image=img)["image"]
        # get output as channels first
        # plt.imshow(img)
        # plt.show()
        metadata = {
            "conditioning_label": 1,
            "class": self.classes[target]
        }
        return img, target, metadata

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    print("fashion mnist loader")

    # unit testing

    # training mode
    dataset = FashionMNIST(data_root="dataset/fashion-mnist",
                           transform=None,
                           mode='train')
    print("training mode", len(dataset))

    for i in range(len(dataset)):
        img, target, metadata = dataset[i]
        print(img.shape)
        print(target, metadata['class'])
        plt.imshow(img)
        plt.show()
    # testing mode
    dataset = FashionMNIST(data_root="dataset/fashion-mnist",
                           transform=None,
                           mode='test')
    print("testing mode", len(dataset))

    for i in range(len(dataset)):
        img, target, metadata = dataset[i]
