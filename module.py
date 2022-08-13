from torch import nn
import torch.nn.functional as F
import torch


# 网络模块接口
class BlockInterface(nn.Module):
    # 参数：输入通道数，输出通道数
    def __init__(self, input_channels, output_channels):
        super(BlockInterface, self).__init__()
        pass

    def forward(self, x):
        pass


# 残差模块
# 参数量：input*output*3*3+output*output*3*3+input*output
class ResidualBlock(BlockInterface):
    def __init__(self, input_channels, output_channels):
        super(ResidualBlock, self).__init__(input_channels, output_channels)
        self.input_channels = input_channels
        self.output_channels = output_channels

        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1)
        # 旁路卷积，卷积核为1，控制通道改变，不对图像自身产生任何变化
        self.conv_side = nn.Conv2d(input_channels, output_channels, kernel_size=1)

    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = self.conv2(y)
        if self.input_channels != self.output_channels:
            x = self.conv_side(x)
        return F.relu(x + y)


# 瓶颈模块
# 参数量：input*low+low*low*3*3+low*output+(input*output)
class BottleneckBlock(BlockInterface):
    def __init__(self, input_channels, output_channels):
        super(BottleneckBlock, self).__init__(input_channels, output_channels)
        self.input_channels = input_channels
        self.output_channels = output_channels
        low_channels = output_channels // 4  # 默认low_channels是output_channels的四分之一

        self.conv1 = nn.Conv2d(input_channels, low_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(low_channels, low_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(low_channels, output_channels, kernel_size=1)
        # 旁路卷积，卷积核为1，控制通道改变，不对图像自身产生任何变化
        self.conv_side = nn.Conv2d(input_channels, output_channels, kernel_size=1)

    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = F.relu(self.conv2(y))
        y = self.conv3(y)
        if self.input_channels != self.output_channels:
            x = self.conv_side(x)
        return F.relu(x + y)


# SE模块
class SEBlock(BlockInterface):
    def __init__(self, input_channels, output_channels):
        super(SEBlock, self).__init__(input_channels, output_channels)
        self.input_channels = input_channels
        self.output_channels = output_channels
        low_channels = output_channels // 4  # 默认low_channels是output_channels的四分之一

        self.conv1 = nn.Conv2d(input_channels, low_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(low_channels, low_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(low_channels, output_channels, kernel_size=1)
        # 旁路卷积，卷积核为1，控制通道改变，不对图像自身产生任何变化
        self.conv_side = nn.Conv2d(input_channels, output_channels, kernel_size=1)
        # SE支路
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(output_channels,output_channels//16,kernel_size=1),  # // 整数除法，返回商的向下取整结果
            nn.ReLU(),
            nn.Conv2d(output_channels//16,output_channels,kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_shortcut = x
        x1 = self.conv1(x)
        x1 = self.conv2(x1)
        x1 = self.conv3(x1)
        x2 = self.se(x1)
        x1 = x1*x2
        if self.input_channels != self.output_channels:
            x_shortcut = self.conv_side(x_shortcut)
        x1 = x1 + x_shortcut
        x1 = F.relu(x1)
        return x1


# Unet模块
class UnetBlock(BlockInterface):
    # channel_factor: 通道变化因子
    # sample_factor: 上下采样因子
    # single_layers: 单个采样臂（上采样臂或下采样臂）的层数（Unet总层数=single_layers×2）
    def __init__(self, input_channels, output_channels, channel_factor, sample_factor, single_layers):
        super(UnetBlock, self).__init__(input_channels, output_channels)
        self.channel_factor = channel_factor
        self.sample_factor = sample_factor
        self.single_layers = single_layers
        self.layers = []
        cur_in = input_channels
        # 下采样
        for i in range(single_layers):
            cur_out = cur_in * 2
            self.layers.append(DownSampleBlock(cur_in, cur_out))
            cur_in = cur_out
        # 上采样
        for i in range(single_layers):
            cur_out = cur_in // 2
            self.layers.append(UpSampleBlock(cur_in, cur_out, sample_factor))
            cur_in = cur_out
        self.outputConv = BottleneckBlock(input_channels, output_channels)

    # 加不加激活函数啊！
    def forward(self, x):
        mid_output = [x]
        i = 0
        for layer in self.layers:
            last_output = mid_output[len(mid_output) - 1]  # 上一层输出
            if i <= self.single_layers - 1:
                mid_output.append(layer(last_output))
            else:
                equal_output = mid_output[self.single_layers * 2 + 1 - i]  # 上采样层的等效层的输出
                mid_output.append(layer(last_output, equal_output))
            i += 1
        fin_output = mid_output[len(mid_output) - 1]
        return self.outputConv(fin_output)


# 下采样模块
class DownSampleBlock(BlockInterface):
    def __init__(self, in_channels, out_channels):
        super(DownSampleBlock, self).__init__(in_channels, in_channels)
        self.maxpool_conv = nn.Sequential(
            BottleneckBlock(in_channels, out_channels),
            BottleneckBlock(out_channels, out_channels),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


# 上采样模块
class UpSampleBlock(nn.Module):
    # sample_factor: 下采样比例，输入图像边长除以此数得到输出图像边长
    # mode: 下采样模式，linear, bilinear, bicubic, trilinear, nearest, Transposed
    def __init__(self, in_channels, out_channels, sample_factor,mode='Transposed'):
        super(UpSampleBlock, self).__init__()
        if mode == 'Transposed':
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = nn.Sequential(
                BottleneckBlock(in_channels, out_channels),
                BottleneckBlock(out_channels, out_channels)
            )
        else:
            self.up = nn.Upsample(scale_factor=sample_factor, mode=mode, align_corners=True)
            self.conv = nn.Sequential(
                BottleneckBlock(in_channels, out_channels),
                BottleneckBlock(out_channels, out_channels)
            )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

