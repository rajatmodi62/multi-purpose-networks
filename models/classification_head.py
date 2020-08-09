import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassificationHead(nn.Module):

    def __init__(self,\
                num_classes=10,\
                in_channels=512):
        super(ClassificationHead, self).__init__()

        self.fc_layer= nn.Linear(in_channels, num_classes)

    def forward(self,x):
        x= self.fc_layer(x)
        return x

if __name__ == '__main__':
    model= ClassificationHead()
    x= torch.randn(4,512)
    output= model(x)
    print("output shape",output.shape)