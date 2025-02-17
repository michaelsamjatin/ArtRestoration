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
  

class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, bias=True):
        super().__init__()
        # The depthwise conv is basically just a grouped convolution in PyTorch with
        # the number of distinct groups being the same as the number of input (and output)
        # channels for that layer.
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, bias=bias, groups=in_channels)
        # The pointwise convolution stretches across all the output channels using
        # a 1x1 kernel.
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
    

class DoubleDSConv(nn.Module):
    def __init__(self, in_channels, out_channels):
      super().__init__()
      self.double_conv = nn.Sequential(
         DepthwiseSeparableConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
         nn.BatchNorm2d(out_channels),
         nn.ReLU(),
         DepthwiseSeparableConv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
         nn.BatchNorm2d(out_channels),
         nn.ReLU()
      )

    def forward(self, x):
      return self.double_conv(x)
    

class DownDSConv2(nn.Module):   
   def __init__(self, in_channels, out_channels, kernel_size=3):
      self.seq = nn.Sequential(
         DepthwiseSeparableConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
         nn.BatchNorm2d(out_channels),
         nn.ReLU(),
         DepthwiseSeparableConv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
         nn.BatchNorm2d(out_channels),
         nn.ReLU()
      )

      self.mp = nn.MaxPool2d(kernel_size=2, return_indices=True)

   def forward(self, x):
      y = self.seq(x)
      pool_shape = y.shape
      y, indices = self.mp(y)
      return y, indices, pool_shape
   
class UpDSConv2(nn.Module):
   def __init__(self, in_channels, out_channels, kernel_size=3):
      super().__init__()
      self.seq = nn.Sequential(
         DepthwiseSeparableConv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
         nn.BatchNorm2d(in_channels),
         nn.ReLU(),
         DepthwiseSeparableConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
         nn.BatchNorm2d(out_channels),
         nn.ReLU()
      )
      self.mup = nn.MaxUnpool2d(kernel_size=2)

   def forward(self, x, indices, output_size):
      y = self.mup(x, indices, output_size=output_size)
      y = self.seq(y)
      return y

