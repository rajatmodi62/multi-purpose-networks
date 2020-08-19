import torch
from torch.utils.data.dataset import ConcatDataset
from dataloader.cifar10 import CIFAR10
from dataloader.fashion_mnist import FashionMNIST
from dataloader.multi_task_batch_scheduler import BatchSchedulerSampler
import matplotlib.pyplot as plt

cifar = CIFAR10(data_root="dataset/cifar10",
                transform=None,
                mode='train',
                )

fashion_mnist = FashionMNIST(data_root="dataset/fashion-mnist",
                             transform=None,
                             mode='train',
                             )


concat_dataset = ConcatDataset([cifar, fashion_mnist])

batch_size = 4

# dataloader with BatchSchedulerSampler
dataloader = torch.utils.data.DataLoader(dataset=concat_dataset,
                                         sampler=BatchSchedulerSampler(dataset=concat_dataset,
                                                                       batch_size=batch_size),
                                         batch_size=batch_size,
                                         )

for inputs in dataloader:
    # print(type(inputs),len(inputs),type(inputs[0]),type(inputs[1]),type(inputs[2]),inputs[0].shape)
    print(inputs[2]['conditioning_label'])
    plt.imshow(inputs[0][0])
    plt.show()
