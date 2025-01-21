import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import DownDSConv2, UpDSConv2

class SegDSCNet(nn.Module):
    '''
    Class for image segmentation - Damage Detection
    '''
    def __init__(self, in_channels, out_channels, filters=[32,64,128,256]):
        super().__init__()

        # normalize input using batch normalization
        self.bn_input = nn.BatchNorm2d(in_channels)
        
        # encoder pathway
        self.dc1 = DownDSConv2(in_channels, filters[0], kernel_size=3)
        self.dc2 = DownDSConv2(filters[0], filters[1], kernel_size=3)
        self.dc3 = DownDSConv2(filters[1], filters[2], kernel_size=3)
        self.dc4 = DownDSConv2(filters[2], filters[3], kernel_size=3)

        self.uc4 = UpDSConv2(filters[3], filters[2], kernel_size=3)
        self.uc3 = UpDSConv2(filters[2], filters[1], kernel_size=3)
        self.uc2 = UpDSConv2(filters[1], filters[0], kernel_size=3)
        self.uc1 = UpDSConv2(filters[0], out_channels, kernel_size=3)

    def forward(self, x):
        x = self.bn_input(x)

        # Encoder
        x, mp1_indices, shape1 = self.dc1(x)
        x, mp2_indices, shape2 = self.dc2(x)
        x, mp3_indices, shape3 = self.dc3(x)
        x, mp4_indices, shape4 = self.dc4(x)

        # Decoder
        x = self.uc4(x, mp4_indices, output_size=shape4)
        x = self.uc3(x, mp3_indices, output_size=shape3)
        x = self.uc2(x, mp2_indices, output_size=shape2)
        x = self.uc1(x, mp1_indices, output_size=shape1)

        return x
