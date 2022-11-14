from torch import nn
import torch.nn.functional as F
from model import module
from model import net
import torch


class Unet(nn.Module):
    def __init__(self, BN=True, dropout=0):
        super().__init__()  # Net -> CNN
        self.intput_conv = nn.Sequential(
            module.BottleneckBlock(1, 32, BN=BN),
            module.BottleneckBlock(32, 32, BN=BN),
            module.BottleneckBlock(32, 64, BN=BN),
            module.BottleneckBlock(64, 64, BN=BN),
            module.BottleneckBlock(64, 128, BN=BN),
            module.BottleneckBlock(128, 128, BN=BN),
            module.BottleneckBlock(128, 128, BN=BN),
            module.BottleneckBlock(128, 64, BN=BN),
            module.BottleneckBlock(64, 64, BN=BN)
        )

        self.unet_block = net.unet(64, 64, 2, 2, 4, 
            BN=True, downsample_mode='Max',upsample_mode='Transposed')

        self.output_conv = nn.Sequential(
            module.BottleneckBlock(64, 32, BN=BN),
            module.BottleneckBlock(32, 32, BN=BN),
            module.BottleneckBlock(32, 32, BN=BN),
            module.ResidualBlock(32, 1, BN=BN)
        )

    def forward(self, x):
        x = self.intput_conv(x)
        x = self.unet_block(x)
        x = self.output_conv(x)
        return x


# 这东西根本不好使，学习率大小都不行
class CascadeUnet(nn.Module):
    def __init__(self, BN=True):
        super().__init__()  # Net -> CNN
        self.unet_block1 = Unet(BN=BN)
        self.unet_block2 = Unet(BN=BN)
        self.unet_block3 = Unet(BN=BN)

    def forward(self, x):
        x = self.unet_block1(x)
        x = self.unet_block2(x)
        x = self.unet_block3(x)
        return x