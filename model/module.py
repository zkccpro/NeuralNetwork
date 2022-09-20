# 所有模块的基础卷积单元都加上了`.to(globalParam.device)`
# 理论上说没有加的必要，因为在执行网络前，推理器会先递归加载所有的网络模块并把他们放到device上。
# 但推理器只能识别各个__init__函数中的`self.conv = conv2d(xxx)`这种形式的网络模块。
# 如果你使用了python list来存放这些卷积，推理器是不是识别并加载的！
# 所以，保险起见，最好为所有模块的基础卷积都加上`.to(globalParam.device)`
from torch import nn
import torch.nn.functional as F
import torch
from conf import globalParam


# 网络块接口
class BlockInterface(nn.Module):
    # 参数：输入通道数，输出通道数，是否开启BN
    # BN开启的话其超参就必须加入反向传播
    # Dropout接受一个0-1的浮点数，0为不开启Dropout，其他为指定Dropout概率
    def __init__(self, input_channels, output_channels,
                BN=True, dropout=0):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.BN = BN

    def forward(self, x):
        pass


# 卷积层封装，支持BN
class Conv(BlockInterface):
    def __init__(self, input_channels, output_channels, 
                kernel_size=3, BN=True):
        super().__init__(input_channels, output_channels, BN=BN)
        pad_size = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, padding=pad_size).to(globalParam.device)
        if self.BN:
            self.bn = nn.BatchNorm2d(output_channels).to(globalParam.device)

    def forward(self, x):
        y = self.conv(x)
        if self.BN:
            y = self.bn(y)
        return y


# 单个线性层封装，支持Dropout
class LinearLayer(BlockInterface):
    def __init__(self, input, output, dropout=0):
        super().__init__(input, output, dropout=dropout)
        self.linear = nn.Sequential(
            nn.Linear(input, output).to(globalParam.device),
            nn.ReLU(),
            nn.Dropout(dropout)
        ) if dropout else nn.Sequential(
            nn.Linear(input, output).to(globalParam.device),
            nn.ReLU()
        )
    def forward(self, x):
        return self.linear(x)


# 全连接层封装，支持Dropout
class FullConnection(BlockInterface):
    def __init__(self, input, output, layer_num=2, dropout=0):
        super().__init__(input, output, dropout=dropout)
        self.layer_num = layer_num
        if self.layer_num > 3:
            print("WARNING: too many layers in full connection")
        if self.layer_num <= 0:
            print('ERROR: layer_num of FullConnection must > 0')
            raise ValueError("layer_num of FullConnection must > 0!")


        # 中间层规模
        mid = 0
        if output > 100:
            mid = output if output < input else input
        elif output <= 100 and output > 10:
            mid = output * 10 if (output * 10) < input else input
        else:  # output <= 10
            mid = output * 100 if (output * 100) < input else input

        # 全连接层，注意最后一层没有Dropout 和 激活函数
        self.fc_layers = []
        if layer_num == 1:
            self.fc_layers.append(nn.Linear(input, output).to(globalParam.device))
        elif layer_num == 2:
            self.fc_layers.append(LinearLayer(input, mid, dropout=dropout))
            self.fc_layers.append(nn.Linear(mid, output).to(globalParam.device))
        else:  # layer_num >= 3
            for i in range(layer_num): # start with 0
                if i == 0:
                    self.fc_layers.append(LinearLayer(input, input, dropout=dropout))
                elif i == 1:
                    self.fc_layers.append(LinearLayer(input, mid, dropout=dropout))
                elif i < (layer_num - 1):
                    self.fc_layers.append(LinearLayer(mid, mid, dropout=dropout))
                else:  # i = layer_num - 1（最后一层）
                    self.fc_layers.append(nn.Linear(mid, output).to(globalParam.device))
    
    def forward(self, x):
        for layer in self.fc_layers:
            x = layer(x)
        return x


# 残差模块
# 参数量：input*output*3*3+output*output*3*3+input*output
class ResidualBlock(BlockInterface):
    def __init__(self, input_channels, output_channels, 
                BN=True):
        super().__init__(input_channels, output_channels,
                        BN=BN)

        self.conv1 = Conv(input_channels, output_channels, BN=BN)
        self.conv2 = Conv(output_channels, output_channels, BN=BN)

        # 旁路卷积，卷积核为1，控制通道改变，只对输入各像素产生整体线性变化
        self.conv_side = Conv(input_channels, output_channels, BN=False, kernel_size=1)

    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = self.conv2(y)
        if self.input_channels != self.output_channels:
            x = self.conv_side(x)
        return F.relu(x + y)


# 瓶颈模块
# 参数量：input*low+low*low*3*3+low*output+(input*output)
# 输出通道不为1，为1时考虑用residual block
class BottleneckBlock(BlockInterface):
    def __init__(self, input_channels, output_channels, 
                BN=True):
        super().__init__(input_channels, output_channels,
                        BN=BN)
        low_channels = output_channels // 4  # 默认low_channels是output_channels的四分之一

        self.conv1 = Conv(input_channels, low_channels, BN=BN, kernel_size=1)
        self.conv2 = Conv(low_channels, low_channels, BN=BN)
        self.conv3 = Conv(low_channels, output_channels, BN=BN, kernel_size=1)

        # 旁路卷积，卷积核为1，控制通道改变，只对输入各像素产生整体线性变化
        self.conv_side = Conv(input_channels, output_channels, BN=False, kernel_size=1)

    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = F.relu(self.conv2(y))
        y = self.conv3(y)
        if self.input_channels != self.output_channels:
            x = self.conv_side(x)
        return F.relu(x + y)


# SE模块
class SEBlock(BlockInterface):
    def __init__(self, input_channels, output_channels, 
                BN=True):
        super().__init__(input_channels, output_channels,
                        BN=BN)
        low_channels = output_channels // 4  # 默认low_channels是output_channels的四分之一

        self.conv1 = Conv(input_channels, low_channels, BN=BN, kernel_size=1)

        self.conv2 = Conv(low_channels, low_channels, BN=BN)

        self.conv3 = Conv(low_channels, output_channels, BN=BN, kernel_size=1)

        # 旁路卷积，卷积核为1，控制通道改变，只对输入各像素产生整体线性变化
        self.conv_side = Conv(input_channels, output_channels, BN=False, kernel_size=1)

        # SE支路
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(output_channels, output_channels//16, kernel_size=1),  # // 整数除法，返回商的向下取整结果
            nn.ReLU(),
            nn.Conv2d(output_channels//16, output_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x1 = F.relu(self.conv2(x1))
        x1 = F.relu(self.conv3(x1))
        x2 = self.se(x1)
        x1 = x1 * x2
        if self.input_channels != self.output_channels:
            x = self.conv_side(x)
        x1 = x1 + x
        return F.relu(x1)


# Unet模块
class UnetBlock(BlockInterface):
    # channel_factor: 通道变化因子
    # sample_factor: 上下采样因子
    # single_layers: 单个采样臂（上采样臂或下采样臂）的层数（Unet总层数=single_layers×2）
    def __init__(self, input_channels, output_channels, 
                channel_factor, sample_factor, single_layers,
                BN=True, downsample_mode='Max',upsample_mode='Transposed'):
        super().__init__(input_channels, output_channels,
                        BN=BN)
        self.channel_factor = channel_factor
        self.sample_factor = sample_factor
        self.single_layers = single_layers
        self.layers = []
        self.mid_output = [None] * (single_layers * 2 + 1)  # n layers have n+1 mid_output
        self.input_channels = input_channels
        self.output_channels = output_channels
        cur_in = input_channels
        # 下采样
        for i in range(single_layers):
            cur_out = cur_in * 2
            self.layers.append(DownSampleBlock(cur_in, cur_out, sample_factor, mode=downsample_mode, BN=BN))
            cur_in = cur_out
        # 上采样
        for i in range(single_layers):
            cur_out = cur_in // 2
            self.layers.append(UpSampleBlock(cur_in, cur_out, sample_factor, BN=BN, mode=upsample_mode))
            cur_in = cur_out
        self.outputConv = BottleneckBlock(input_channels, output_channels, BN=BN)

    def forward(self, x):
        i = 0
        self.mid_output[0] = x
        for i in range(len(self.layers)):
            last_output = self.mid_output[i]  # 上一层输出
            if i < self.single_layers:
                self.mid_output[i + 1] = self.layers[i](last_output)
            else:
                equal_output = self.mid_output[self.single_layers * 2 - i - 1]  # 上采样层的等效层的输出
                self.mid_output[i + 1] = self.layers[i](last_output, equal_output)
            i += 1
        fin_output = self.mid_output[len(self.mid_output) - 1]
        if self.input_channels != self.output_channels:
            fin_output = self.outputConv(fin_output)
        return fin_output


# 下采样模块
class DownSampleBlock(BlockInterface):
    # mode: 降采样类型，Max or Avg
    def __init__(self, in_channels, output_channels, sample_factor,
                mode='Max', BN=True):
        super().__init__(in_channels, output_channels, BN=BN)
        self.conv = BottleneckBlock(in_channels, output_channels, BN=BN)
        self.pool = nn.AvgPool2d(sample_factor).to(globalParam.device) if mode == 'Avg' else nn.MaxPool2d(sample_factor).to(globalParam.device)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x


# 上采样模块
class UpSampleBlock(BlockInterface):
    # sample_factor: 下采样比例，输入图像边长除以此数得到输出图像边长
    # mode: 下采样模式，linear, bilinear, bicubic, trilinear, nearest, Transposed
    def __init__(self, in_channels, output_channels, 
                sample_factor,mode='Transposed', BN=True):
        super().__init__(in_channels, output_channels,
                        BN=BN)
        if mode == 'Transposed':
            self.up = nn.ConvTranspose2d(in_channels, output_channels, kernel_size=2, stride=2).to(globalParam.device)
            self.conv = BottleneckBlock(in_channels, output_channels, BN=BN)
        else:
            self.up = nn.Upsample(scale_factor=sample_factor, mode=mode, align_corners=True).to(globalParam.device)
            self.conv = BottleneckBlock(in_channels, output_channels, BN=BN)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # # input is CHW
        # diffY = x2.size()[2] - x1.size()[2]
        # diffX = x2.size()[3] - x1.size()[3]
        #
        # x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
        #                 diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

