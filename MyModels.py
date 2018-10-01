import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class AutoEncoder(nn.Module):
    """
    AutoEncoder module 
    """
    def __init__(self,dropout_rate):
        super(AutoEncoder, self).__init__()

        #Encoder is a RenNet18
        pretrained=models.resnet18(pretrained=True)

        pretrained.avgpool=nn.AdaptiveAvgPool2d(1) 

        #Remove the last  fc layer

        self.encoder= nn.Sequential(*list(pretrained.children())[:-1]) 

        self.decoder=Decoder()

        #Dropout on the compressed encoding
        self.dropout=nn.Dropout2d(dropout_rate,inplace=True)


    def forward(self,x):
        """
        Args:
            x:    [N,C,H,W] input 4D Pytorch tensor
        """
        #Compressed representation

        y=self.encoder(x)

        #Apply dropout in compressed encoding during training

        y=self.dropout(y)

        #Decode
        y=self.decoder(y)

        return y

# Create Nearest Neighbour Up-sampling module to be used in the decoder

class NearestUsampling2D(nn.Module):
    def __init__(self,size):
        super(NearestUsampling2D, self).__init__()
        self.size=size #(tuple)

    def forward(self,input):
        return F.interpolate(input, size=self.size,mode='nearest')


#Set up decoder
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.decoder=nn.Sequential(
            #1st dconv layer
            nn.BatchNorm2d(512), # [N,input_dims,1,1]
            NearestUsampling2D((2,2)), # [N,input_dims,2,2]
            nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1), # [N,input_dims,2,2]
            nn.BatchNorm2d(512),
            nn.RReLU(),

            #2nd dconv layer
            NearestUsampling2D((4,4)), # [N,512,4,4]
            nn.Conv2d(512,256,kernel_size=3,stride=1,padding=1), # [N,256,4,4]
            nn.BatchNorm2d(256),
            nn.RReLU(),

            #3rd dconv layer
            NearestUsampling2D((8,8)), # [N,256,8,8]
            nn.Conv2d(256,128,kernel_size=3,stride=1,padding=1), # [N,128,8,8]
            nn.BatchNorm2d(128),
            nn.RReLU(),

            #4th dconv layer
            NearestUsampling2D((16,16)), # [N,128,16,16]
            nn.Conv2d(128,64,kernel_size=3,stride=1,padding=1), # [N,64,16,16]
            nn.BatchNorm2d(64),
            nn.RReLU(),

            #5th dconv layer
            NearestUsampling2D((34,34)), # [N,64,34,34]
            nn.Conv2d(64,32,kernel_size=5,stride=1,padding=1), # [N,32,32,32]
            nn.BatchNorm2d(32),
            nn.RReLU(),

            #6th dconv layer
            NearestUsampling2D((64,64)), # [N,32,64,64]
            nn.Conv2d(32,16,kernel_size=5,stride=1,padding=1),# [N,16,62,62]
            nn.BatchNorm2d(16),
            nn.RReLU(),

            #6th dconv layer
            NearestUsampling2D((98,98)), # [N,16,98,98]
            nn.Conv2d(16,3,kernel_size=5,stride=1,padding=1), # [N,3,96,96]
            nn.Sigmoid())

    def forward(self,x):
         return self.decoder(x)