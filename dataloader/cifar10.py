import os
import numpy as np
import matplotlib.pyplot as plt
import glob as glob
from PIL import Image
import torch
import torch.utils.data as data
import pickle
from pathlib import Path
from utils.augmentation import get_cifar_augmentation

# Classes for this dataset are:
# 'airplane': 0
# 'automobile': 1
# 'bird': 2
# 'cat': 3
# 'deer': 4
# 'dog': 5
# 'frog': 6
# 'horse': 7
# 'ship': 8
# 'truck': 9


class CIFAR10(data.Dataset):

    def __init__(self, data_root, transform=None, mode='train'):
        super().__init__()

        if transform is not None:
            self.transform = transform
        else:
            print("fetching default transform")
            self.transform = get_cifar_augmentation(mode=mode)
        self.mode = mode
        self.data_root = data_root
        self.train_list = [
            'data_batch_1',
            'data_batch_2',
            'data_batch_3',
            'data_batch_4',
            'data_batch_5',
        ]

        self.test_list = [
            'test_batch',
        ]

        self.meta = {
            'filename': 'batches.meta',
            'key': 'label_names',
            'md5': '5ff9c542aee3614f3951f8cda6e48888',
        }

        if mode == 'train':
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []
        self.targets = []
        self.local_data_path = Path('.').absolute()

        # iterate through the list
        for file_name in downloaded_list:
            file_path = str(self.local_data_path/self.data_root/file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        self._load_meta()

        self.classes = ['airplane',
                        'automobile',
                        'bird',
                        'cat',
                        'deer',
                        'dog',
                        'frog',
                        'horse',
                        'ship',
                        'truck'
                        ]

    def _load_meta(self):
        path = str(self.local_data_path/self.data_root/self.meta['filename'])
        with open(path, 'rb') as infile:
            data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i,
                             _class in enumerate(self.classes)}
        print("self.class_to_idx", self.class_to_idx)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        # orig_img=img.copy()
        # print("type of image",type(img),img.dtype)
        # print("image shape",img.shape)
        # apply transformation to the image
        img = self.transform(image=img)["image"]
        # print("After,",img.shape)
        # plt.subplot(1,2,1)
        # plt.imshow(orig_img)
        # plt.subplot(1,2,2)
        # plt.imshow(img)
        # plt.show()
        metadata = {
            "conditioning_label": 0,
            "class": self.classes[target]
        }

        return img, target, metadata

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    print("cifar 10 loader")

    # unit testing

    # training mode
    dataset = CIFAR10(data_root="dataset/cifar10",
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
    dataset = CIFAR10(data_root="dataset/cifar10",
                      transform=None,
                      mode='test')
    print("testing mode", len(dataset))

    for i in range(len(dataset)):
        img, target, metadata = dataset[i]
