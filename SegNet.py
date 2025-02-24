import torch.nn as nn
from modules import DownConv2, UpConv2

class SegNet(nn.Module):
  def __init__(self, in_channels, out_channels, filters=[32,64,128,256]):
    super().__init__()

    # normalize input using batch norm
    self.bn_input = nn.BatchNorm2d(in_channels)

    # encoder pathway
    self.dc1 = DownConv2(in_channels, filters[0], kernel_size=3)
    self.dc2 = DownConv2(filters[0], filters[1], kernel_size=3)
    self.dc3 = DownConv2(filters[1], filters[2], kernel_size=3)
    self.dc4 = DownConv2(filters[2], filters[3], kernel_size=3)

    # decoder pathway
    self.uc4 = UpConv2(filters[3], filters[2], kernel_size=3)
    self.uc3 = UpConv2(filters[2], filters[1], kernel_size=3)
    self.uc2 = UpConv2(filters[1], filters[0], kernel_size=3)
    self.uc1 = UpConv2(filters[0], out_channels, kernel_size=3)

  def forward(self, x):
    x = self.bn_input(x)

    # encoder
    x, mp1_indices, shape1 = self.dc1(x)
    x, mp2_indices, shape2 = self.dc2(x)
    x, mp3_indices, shape3 = self.dc3(x)
    x, mp4_indices, shape4 = self.dc4(x)

    # decoder
    x = self.uc4(x, mp4_indices, shape4)
    x = self.uc3(x, mp3_indices, shape3)
    x = self.uc2(x, mp2_indices, shape2)
    x = self.uc1(x, mp1_indices, shape1)

    return x