import torch
import torch.nn as nn

# GENERATOR
class UNetGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, num_filters=64):
        super(UNetGenerator, self).__init__()

        # Encoder
        self.encoder1 = self._block(in_channels, num_filters)
        self.encoder2 = self._block(num_filters, num_filters * 2)
        self.encoder3 = self._block(num_filters * 2, num_filters * 4)
        self.encoder4 = self._block(num_filters * 4, num_filters * 8)

        # Bottleneck
        self.bottleneck = self._block(num_filters * 8, num_filters * 16, dilation=2)

        # Decoder with skip connections
        self.decoder1 = self._block(num_filters * 16 + num_filters * 8, num_filters * 8, dilation=2)
        self.decoder2 = self._block(num_filters * 8 + num_filters * 4, num_filters * 4)
        self.decoder3 = self._block(num_filters * 4 + num_filters * 2, num_filters * 2)
        self.decoder4 = self._block(num_filters * 2 + num_filters, num_filters)

        # Final layer
        self.final = nn.Conv2d(num_filters, out_channels, kernel_size=1)

    def _block(self, in_channels, out_channels, dilation=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)
        e2 = self.encoder2(nn.MaxPool2d(2)(e1))
        e3 = self.encoder3(nn.MaxPool2d(2)(e2))
        e4 = self.encoder4(nn.MaxPool2d(2)(e3))

        # Bottleneck
        bottleneck = self.bottleneck(nn.MaxPool2d(2)(e4))

        # Decoder with skip connections
        d1 = self.decoder1(torch.cat([nn.Upsample(scale_factor=2)(bottleneck), e4], dim=1))
        d2 = self.decoder2(torch.cat([nn.Upsample(scale_factor=2)(d1), e3], dim=1))
        d3 = self.decoder3(torch.cat([nn.Upsample(scale_factor=2)(d2), e2], dim=1))
        d4 = self.decoder4(torch.cat([nn.Upsample(scale_factor=2)(d3), e1], dim=1))

        # Final layer
        return torch.tanh(self.final(d4))
    
# DISCRIMINATOR
class Discriminator(nn.Module):
    def __init__(self, in_channels=3, num_filters=64):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels, num_filters, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_filters, num_filters * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_filters * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_filters * 2, num_filters * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_filters * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_filters * 4, num_filters * 8, kernel_size=4, stride=2, padding=2, dilation=2),
            nn.BatchNorm2d(num_filters * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_filters * 8, 1, kernel_size=16, stride=1, padding=0)
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels=3, num_filters=64):
        super(PatchGANDiscriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels, num_filters, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_filters, num_filters * 2, kernel_size=4, stride=2, padding=2, dilation=2),
            nn.BatchNorm2d(num_filters * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_filters * 2, num_filters * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_filters * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_filters * 4, num_filters * 8, kernel_size=4, stride=1, padding=1),  # No stride
            nn.BatchNorm2d(num_filters * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_filters * 8, 1, kernel_size=4, stride=1, padding=1),  # Output a matrix
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
