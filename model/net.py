
# 按照论文NiN(网络中的网络)的思想先搭建网络中的网络
from model import module as md
from torch import nn
from conf import globalParam


# resnet论文中风格
# nblock: 有n个残差块
# down_sampling: 是否在第一个残差块处降采样（按论文里的风格，用stride=2来降采样）
# block_type: 采用哪种块，现支持'res','neck','se' todo: 这里应该反射
class resnet(md.BlockInterface):
    def __init__(self, input_channels, output_channels, nblock,
                down_sampling=False, block_type='res', BN=True):
        super().__init__(input_channels, output_channels, BN=BN)
        if nblock <= 0:
            print('ERROR: nblock of resnet must > 0!')
            raise ValueError("nblock of resnet must > 0!")
        
        self.blocks = []

        # 首块，按照传参决定是否下采样
        if block_type == 'neck':
            self.blocks.append(md.BottleneckBlock(input_channels, output_channels, down_sampling=down_sampling, BN=BN))
        elif block_type == 'res':
            self.blocks.append(md.ResidualBlock(input_channels, output_channels, down_sampling=down_sampling, BN=BN))
        else: # block_type == 'se'
            self.blocks.append(md.SEBlock(input_channels, output_channels, down_sampling=down_sampling, BN=BN))
        # 其余块，均取默认，不下采样
        for i in range(nblock - 1):
            if block_type == 'neck':
                self.blocks.append(md.BottleneckBlock(output_channels, output_channels, BN=BN))
            elif block_type == 'res':
                self.blocks.append(md.ResidualBlock(output_channels, output_channels, BN=BN))
            else: # block_type == 'se'
                self.blocks.append(md.SEBlock(output_channels, output_channels, BN=BN))

    def forward(self, x):
        i = 0
        for block in self.blocks:
            x = block(x)
            # print(i, x.shape)
            i += 1
        return x


# Unet网络
class unet(md.BlockInterface):
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
            self.layers.append(md.DownSampleBlock(cur_in, cur_out, sample_factor, mode=downsample_mode, BN=BN))
            cur_in = cur_out
        # 上采样
        for i in range(single_layers):
            cur_out = cur_in // 2
            self.layers.append(md.UpSampleBlock(cur_in, cur_out, sample_factor, BN=BN, mode=upsample_mode))
            cur_in = cur_out
        self.outputConv = md.BottleneckBlock(input_channels, output_channels, BN=BN)

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


# 全连接层封装，支持Dropout
class FullConnection(md.BlockInterface):
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
            self.fc_layers.append(md.LinearLayer(input, mid, dropout=dropout))
            self.fc_layers.append(nn.Linear(mid, output).to(globalParam.device))
        else:  # layer_num >= 3
            for i in range(layer_num): # start with 0
                if i == 0:
                    self.fc_layers.append(md.LinearLayer(input, input, dropout=dropout))
                elif i == 1:
                    self.fc_layers.append(md.LinearLayer(input, mid, dropout=dropout))
                elif i < (layer_num - 1):
                    self.fc_layers.append(md.LinearLayer(mid, mid, dropout=dropout))
                else:  # i = layer_num - 1（最后一层）
                    self.fc_layers.append(nn.Linear(mid, output).to(globalParam.device))
    
    def forward(self, x):
        for layer in self.fc_layers:
            x = layer(x)
        return x
