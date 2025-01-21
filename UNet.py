import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import DoubleConv

class UNet(nn.Module):
  def __init__(self, in_channels, out_channels, filters=[8,16,32,64]):
    super().__init__()

    # encoder pathway
    self.conv1 = DoubleConv(in_channels, filters[0])
    self.conv2 = DoubleConv(filters[0], filters[1])
    self.conv3 = DoubleConv(filters[1], filters[2])
    self.conv4 = DoubleConv(filters[2], filters[3])

    # Pooling
    self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    # Decoder Pathway
    self.up1 = nn.ConvTranspose2d(filters[3], filters[2], kernel_size=2, stride=2)
    self.upconv1 = DoubleConv(filters[3], filters[2])

    self.up2 = nn.ConvTranspose2d(filters[2], filters[1], kernel_size=2, stride=2)
    self.upconv2 = DoubleConv(filters[2], filters[1])

    self.up3 = nn.ConvTranspose2d(filters[1], filters[0], kernel_size=2, stride=2)
    self.upconv3 = DoubleConv(filters[1], filters[0])

    # final output
    self.final_conv = nn.Conv2d(filters[0], out_channels, kernel_size=1)

  def forward(self, x):
    # Encoder
    conv1 = self.conv1(x)
    x = self.pool(conv1)

    conv2 = self.conv2(x)
    x = self.pool(conv2)

    conv3 = self.conv3(x)
    x = self.pool(conv3)

    # bridge to decoder
    x = self.conv4(x)

    # Decoder
    x = self.up1(x)

    # concat (is this legit? I mean do the dimensions fit?)
    x = torch.cat([x, conv3], dim=1)
    x = self.upconv1(x)

    x = self.up2(x)
    x = torch.cat([x, conv2], dim=1)
    x = self.upconv2(x)

    x = self.up3(x)
    x = torch.cat([x, conv1], dim=1)
    x = self.upconv3(x)

    # final output
    return self.final_conv(x)