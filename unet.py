import torch
import torch.nn as nn
import torch.nn.functional as F


# Attention Block
class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )   
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
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


# U-Net with Attention Mechanisms
class AttentionUNet(nn.Module):
    def __init__(self, input_channels=4, output_channels=3):
        super(AttentionUNet, self).__init__()

        # Encoder
        self.encoder1 = self.conv_block(input_channels, 64, 1)
        self.encoder2 = self.conv_block(64, 128, 1)
        self.encoder3 = self.conv_block(128, 256, 2)
        self.encoder4 = self.conv_block(256, 512, 2)

        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024, 4)

        # Decoder
        self.upconv4 = self.up_conv(1024, 512)
        self.att4 = AttentionBlock(F_g=512, F_l=512, F_int=256)
        self.decoder4 = self.conv_block(1024, 512, 2)

        self.upconv3 = self.up_conv(512, 256)
        self.att3 = AttentionBlock(F_g=256, F_l=256, F_int=128)
        self.decoder3 = self.conv_block(512, 256, 2)

        self.upconv2 = self.up_conv(256, 128)
        self.att2 = AttentionBlock(F_g=128, F_l=128, F_int=64)
        self.decoder2 = self.conv_block(256, 128, 1)

        self.upconv1 = self.up_conv(128, 64)
        self.att1 = AttentionBlock(F_g=64, F_l=64, F_int=32)
        self.decoder1 = self.conv_block(128, 64, 1)

        # Final Output
        self.final_conv = nn.Conv2d(64, output_channels, kernel_size=1, stride=1, padding=0)

    def conv_block(self, in_channels, out_channels, dilation=2):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),  # Regular Convolution
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation, bias=True),  # Dilated Convolution
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


    def up_conv(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)
        e2 = self.encoder2(F.max_pool2d(e1, kernel_size=2, stride=2))
        e3 = self.encoder3(F.max_pool2d(e2, kernel_size=2, stride=2))
        e4 = self.encoder4(F.max_pool2d(e3, kernel_size=2, stride=2))

        # Bottleneck
        b = self.bottleneck(F.max_pool2d(e4, kernel_size=2, stride=2))

        # Decoder
        d4 = self.upconv4(b)
        e4 = self.att4(g=d4, x=e4)
        d4 = torch.cat((e4, d4), dim=1)
        d4 = self.decoder4(d4)

        d3 = self.upconv3(d4)
        e3 = self.att3(g=d3, x=e3)
        d3 = torch.cat((e3, d3), dim=1)
        d3 = self.decoder3(d3)

        d2 = self.upconv2(d3)
        e2 = self.att2(g=d2, x=e2)
        d2 = torch.cat((e2, d2), dim=1)
        d2 = self.decoder2(d2)

        d1 = self.upconv1(d2)
        e1 = self.att1(g=d1, x=e1)
        d1 = torch.cat((e1, d1), dim=1)
        d1 = self.decoder1(d1)

        # Final Output
        out = self.final_conv(d1)
        return out
