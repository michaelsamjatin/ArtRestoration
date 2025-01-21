import torch
import torch.nn as nn
import torch.nn.functional as F
    

class DoubleConv(nn.Module):
  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.double_conv = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), # using padding = 1 will keep the original spatial dimension
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

  def forward(self, x):
    return self.double_conv(x)


class AttentionGate(nn.Module):
  """Help the model to focus relevant features and suppress irrelevant ones."""
  def __init__(self, F_g, F_l, F_int):
      super().__init__()
      self.W_g = nn.Sequential(
          nn.Conv2d(F_g, F_int, kernel_size=1),
          nn.BatchNorm2d(F_int)
      )
      self.W_x = nn.Sequential(
          nn.Conv2d(F_l, F_int, kernel_size=1),
          nn.BatchNorm2d(F_int)
      )
      self.psi = nn.Sequential(
          nn.Conv2d(F_int, 1, kernel_size=1),
          nn.BatchNorm2d(1),
          nn.Sigmoid()
      )
      self.relu = nn.ReLU(inplace=True)

  def forward(self, g, x):
      g1 = self.W_g(g)
      x1 = self.W_x(x)
      psi = self.relu(g1 + x1)
      psi = self.psi(psi)
      return x * psi


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
  

class AttentionUnet(nn.Module):
  def __init__(self, in_channels, out_channels, filters=[32, 64, 128, 256]):
      super().__init__()
      
      # Encoder pathway
      self.conv1 = DoubleConv(in_channels, filters[0])
      self.conv2 = DoubleConv(filters[0], filters[1])
      self.conv3 = DoubleConv(filters[1], filters[2])
      self.conv4 = DoubleConv(filters[2], filters[3])
      
      # Pooling
      self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

      # Attention Gates
      self.attention1 = AttentionGate(F_g=filters[2], F_l=filters[2], F_int=filters[2]//2)
      self.attention2 = AttentionGate(F_g=filters[1], F_l=filters[1], F_int=filters[1]//2)
      self.attention3 = AttentionGate(F_g=filters[0], F_l=filters[0], F_int=filters[0]//2)
      
      # Decoder pathway
      self.up1 = nn.ConvTranspose2d(filters[3], filters[2], kernel_size=2, stride=2)
      self.upconv1 = DoubleConv(filters[3], filters[2])
      
      self.up2 = nn.ConvTranspose2d(filters[2], filters[1], kernel_size=2, stride=2)
      self.upconv2 = DoubleConv(filters[2], filters[1])
      
      self.up3 = nn.ConvTranspose2d(filters[1], filters[0], kernel_size=2, stride=2)
      self.upconv3 = DoubleConv(filters[1], filters[0])
      
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