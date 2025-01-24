import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(UNet, self).__init__()

        # Encoder
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)

        # Bottleneck
        self.bottleneck = self.conv_block(128, 256)

        # Decoder
        self.dec2 = self.conv_block(256, 128)
        self.dec1 = self.conv_block(128, 64)

        # Final output layer
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels, kernel_size=3, padding=1):
        """Basic convolutional block with 2 conv layers and ReLU activations"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, t):
        # Apply simple time embedding
        t_embedding = torch.sin(t.float()).view(-1, 1, 1, 1)
        x = x + t_embedding 

        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)

        # Bottleneck
        bottleneck = self.bottleneck(enc2)

        # Decoder
        dec2 = self.dec2(bottleneck)
        dec1 = self.dec1(dec2)

        out = self.final_conv(dec1)
        return out