import torch
import torch.nn as nn
import torch.nn.functional as F

from models.resnet import resnet18


#Hardcode in_channels if channels other than 3/4 are passed
# To do: Load all the weights except first layer 
class Backbone(nn.Module):
    def __init__(self,\
                backbone='resnet18',\
                apply_embedding= False,\
                in_channels= None,\
                pretrained= False):
        super(Backbone, self).__init__()
        
        self.apply_embedding= apply_embedding

        if in_channels:
            print("in channels",in_channels)
            self.in_channels=in_channels 
        elif apply_embedding:
            self.in_channels= 4
        else:
            self.in_channels= 3
        #load model
        if backbone=='resnet18':
            self.backbone= resnet18(in_channels=self.in_channels)
        #define the embedding layer for now 
        if apply_embedding:
            self.embedding= nn.Embedding(2,32*32*1)

    def forward(self,img,embedding_label=None):
        
        #fwd pass through embedding layer 
        if embedding_label!=None:
            embedding_vector= self.embedding(embedding_label)
            embedding_vector= embedding_vector.view(-1,1,32,32)
            #concatenate embedding vector to img along channel dimension 
            img = torch.cat([img,embedding_vector],dim=1)
        #fwd pass through backbone 
        x = self.backbone(img)
        return x

if __name__ == '__main__':
    model= Backbone(apply_embedding=True)
    img= torch.randn(8,3,32,32)
    embedding_label= torch.tensor([[0],[1],[1],[0],[0],[0],[0],[0]])
    
    print("img shape",img.shape)
    print("embedding shape",embedding_label.shape)
    output= model(img,embedding_label)
    print("output shape",output.shape)