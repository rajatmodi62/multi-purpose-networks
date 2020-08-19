import os
import numpy as np
import matplotlib.pyplot as plt
import glob as glob
from PIL import Image
import torch
import torch.utils.data as data
import pickle
from pathlib import Path
from utils.augmentation import get_cifar_fashmnist_augmentation


class CIFAR_FASHMNIST(data.Dataset):

    def __init__(self, cifar_data_root, fashion_mnist_data_root, transform=None, mode='train'):
        super().__init__()
        print("combined dataset called")
        if transform is not None:
            self.transform = transform
        else:
            print("fetching default transform")
            self.transform = get_cifar_fashmnist_augmentation(mode=mode)

        # define classes

        self.cifar_classes = [
            'airplane',
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
        self.fashion_mnist_classes = [
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

        self.classes = self.cifar_classes + self.fashion_mnist_classes

        # get the cifar data
        self.cifar_data, self.cifar_targets = self.gather_cifar_data(
            cifar_data_root, mode=mode)
        # get fashion_mnist_data
        self.fashion_mnist_data, self.fashion_mnist_targets = self.gather_fashion_mnist_data(
            fashion_mnist_data_root, mode=mode)

        # generate combined dataset
        self.data = self.cifar_data + self.fashion_mnist_data
        # rescale fashion mnist targets according to combined scales
        self.fashion_mnist_targets = [
            target+len(self.cifar_classes) for target in self.fashion_mnist_targets]
        self.targets = self.cifar_targets + self.fashion_mnist_targets

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        if torch.is_tensor(img):
            img = img.numpy()
        img = self.transform(image=img)["image"]
        metadata = {
            "conditioning_label": 2,
            "class": self.classes[target]
        }

        return img, target, metadata

    def __len__(self):
        return len(self.data)

    def gather_cifar_data(self, cifar_data_root, mode='train'):
        train_list = [
            'data_batch_1',
            'data_batch_2',
            'data_batch_3',
            'data_batch_4',
            'data_batch_5',
        ]
        test_list = [
            'test_batch',
        ]

        if mode == 'train':
            downloaded_list = train_list
        else:
            downloaded_list = test_list
        data = []
        targets = []
        local_data_path = Path('.').absolute()

        # iterate through the list
        for file_name in downloaded_list:
            file_path = str(local_data_path/cifar_data_root/file_name)

            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                data.append(entry['data'])
                if 'labels' in entry:
                    targets.extend(entry['labels'])
                else:
                    targets.extend(entry['fine_labels'])

        data = np.vstack(data).reshape(-1, 3, 32, 32)
        data = data.transpose((0, 2, 3, 1))  # convert to HWC
        # convert data to a list of numpy arrays of 32X32X3
        data = [data[i] for i in range(data.shape[0])]
        return data, targets

    def gather_fashion_mnist_data(self, fashion_mnist_data_root, mode="train"):
        local_data_path = Path('.').absolute()
        if mode == 'train':
            data_file = 'processed/training.pt'
        else:
            data_file = 'processed/test.pt'
        data, targets = torch.load(
            str(local_data_path/fashion_mnist_data_root/data_file))
        # repeat image along three axis for compatible cnn input
        data = [np.repeat(img[:, :, np.newaxis], 3, axis=2) for img in data]
        return data, targets


if __name__ == '__main__':
    dataset = CIFAR_FASHMNIST(cifar_data_root="dataset/cifar10",
                              fashion_mnist_data_root="dataset/fashion-mnist",
                              transform=None,
                              mode='train')

    print("training mode", len(dataset))

    for i in range(len(dataset)):
        if i < 70000:
            continue
        img, target, metadata = dataset[i]
        # print(img.shape)
        print(img.shape, target, metadata['class'])
        plt.imshow(img)
        plt.show()
    # testing mode
    dataset = CIFAR_FASHMNIST(cifar_data_root="dataset/cifar10",
                              fashion_mnist_data_root="dataset/fashion-mnist",
                              transform=None,
                              mode='test')
    print("testing mode", len(dataset))

    for i in range(len(dataset)):
        img, target, metadata = dataset[i]
