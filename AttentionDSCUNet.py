import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import DoubleDSConv, AttentionGate

class AttentionUnet(nn.Module):
  def __init__(self, in_channels, out_channels, filters=[32, 64, 128, 256]):
      super().__init__()
      
      # Encoder pathway
      self.conv1 = DoubleDSConv(in_channels, filters[0])
      self.conv2 = DoubleDSConv(filters[0], filters[1])
      self.conv3 = DoubleDSConv(filters[1], filters[2])
      self.conv4 = DoubleDSConv(filters[2], filters[3])
      
      # Pooling
      self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

      # Attention Gates
      self.attention1 = AttentionGate(F_g=filters[2], F_l=filters[2], F_int=filters[2]//2)
      self.attention2 = AttentionGate(F_g=filters[1], F_l=filters[1], F_int=filters[1]//2)
      self.attention3 = AttentionGate(F_g=filters[0], F_l=filters[0], F_int=filters[0]//2)
      
      # Decoder pathway
      self.up1 = nn.ConvTranspose2d(filters[3], filters[2], kernel_size=2, stride=2)
      self.upconv1 = DoubleDSConv(filters[3], filters[2])
      
      self.up2 = nn.ConvTranspose2d(filters[2], filters[1], kernel_size=2, stride=2)
      self.upconv2 = DoubleDSConv(filters[2], filters[1])
      
      self.up3 = nn.ConvTranspose2d(filters[1], filters[0], kernel_size=2, stride=2)
      self.upconv3 = DoubleDSConv(filters[1], filters[0])
      
      # Final output
      self.final_conv = nn.Conv2d(filters[0], out_channels, kernel_size=1)
      
  def forward(self, x):
      # Encoder
      conv1 = self.conv1(x)
      x = self.pool(conv1)
      
      conv2 = self.conv2(x)
      x = self.pool(conv2)
      
      conv3 = self.conv3(x)
      x = self.pool(conv3)
      
      # Bridge to decoder
      x = self.conv4(x)
      
      # Decoder with attention gates
      x = self.up1(x)
      conv3_att = self.attention1(g=x, x=conv3)
      # Concatenate to combine fine spatial details with high-level information
      x = torch.cat([x, conv3_att], dim=1)
      x = self.upconv1(x)

      x = self.up2(x)
      conv2_att = self.attention2(g=x, x=conv2)
      x = torch.cat([x, conv2_att], dim=1)
      x = self.upconv2(x)

      x = self.up3(x)
      conv1_att = self.attention3(g=x, x=conv1)
      x = torch.cat([x, conv1_att], dim=1)
      x = self.upconv3(x)

      
      # Final output
      return self.final_conv(x)