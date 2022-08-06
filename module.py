from torch import nn
import torch.nn.functional as F


# 残差模块
class ResidualBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(ResidualBlock, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels

        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1)
        # 旁路卷积，卷积核为1，控制通道改变，不对图像自身产生任何变化
        self.conv_side = nn.Conv2d(input_channels, output_channels, kernel_size=1)

    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = self.conv2(y)
        x = self.conv_side(x)
        return F.relu(x + y)


# 瓶颈模块
class BottleneckBlock(nn.Module):
    def __init__(self, input_channels, output_channels, low_channels):
        super(BottleneckBlock, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.low_channels = low_channels

        self.conv1 = nn.Conv2d(input_channels, low_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(low_channels, low_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(low_channels, output_channels, kernel_size=1)
        # 旁路卷积，卷积核为1，控制通道改变，不对图像自身产生任何变化
        self.conv_side = nn.Conv2d(input_channels, output_channels, kernel_size=1)

    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = F.relu(self.conv2(y))
        y = self.conv3(y)
        x = self.conv_side(x)
        return F.relu(x + y)