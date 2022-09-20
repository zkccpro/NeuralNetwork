from torch import nn
import torch.nn.functional as F
from model import module
import torch


class ResidualCNN(nn.Module):
    def __init__(self, BN=True, dropout=0):
        super().__init__()  # Net -> CNN
        # Conv2d: input_channels, output_channels, kernel_size
        # param_num: input_channels * output_channels * kernel_size * kernel_size
        # padding是在外围补充的圈数，padding=1就补1圈，为了和原图尺寸保持一致（别忘了，补1圈，边长+2！）
        # 如果不加padding的话，卷积后图像尺寸变化为：out_size = in_size - kernel_size + 1
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.residual1 = module.ResidualBlock(64, 128, BN=BN)
        self.residual2 = module.ResidualBlock(128, 64, BN=BN)
        self.residual3 = module.ResidualBlock(64, 64, BN=BN)

        self.fc = module.FullConnection(1600, 1, layer_num=2, dropout=dropout)

    def forward(self, x):
        input_size = x.size(0)
        # print(x.shape)
        # in: batch*1*240*240, out: batch*64*238*238
        x = self.conv1(x)
        # print(x.shape)
        # out: batch*64*240*240
        x = F.relu(x)
        # in: batch*64*240*240, out: batch*64*120*120
        x = F.max_pool2d(x, 2, 2)
        # print(x.shape)

        # in: batch*64*120*120, out: batch*128*120*120
        x = self.residual1(x)
        # print(x.shape)
        x = F.relu(x)
        # in: batch*128*120*120, out: batch*128*40*40
        x = F.max_pool2d(x, 3, 3)
        # print(x.shape)

        # in: batch*128*40*40, out: batch*64*40*40
        x = self.residual2(x)
        # print(x.shape)
        x = F.relu(x)
        # in: batch*64*40*40, out: batch*64*20*20
        x = F.max_pool2d(x, 2, 2)
        # print(x.shape)

        # in: batch*64*20*20, out: batch*64*20*20
        x = self.residual3(x)
        # print(x.shape)
        x = F.relu(x)
        # in: batch*64*20*20, out: batch*64*5*5
        x = F.avg_pool2d(x, 4, 4)
        # print(x.shape)

        # batch*64*5*5
        x = x.view(input_size, -1)
        # print(x.shape)
        # in: batch*64*5*5=batch*1600  out:batch*200

        return self.fc(x)


class BottleneckCNNDoubleInput(nn.Module):
    def __init__(self, BN=True, dropout=0):
        super().__init__()  # Net -> CNN
        # Conv2d: input_channels, output_channels, kernel_size
        # param_num: input_channels * output_channels * kernel_size * kernel_size
        # padding是在外围补充的圈数，padding=1就补1圈，为了和原图尺寸保持一致（别忘了，补1圈，边长+2！）
        # 如果不加padding的话，卷积后图像尺寸变化为：out_size = in_size - kernel_size + 1
        self.conv_big = nn.Conv2d(1, 32, kernel_size=7, padding=3)
        self.residual0 = module.BottleneckBlock(32, 32, BN=BN)

        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.residual1 = module.BottleneckBlock(64, 64, BN=BN)
        self.residual2 = module.BottleneckBlock(64, 64, BN=BN)

        self.residual3 = module.BottleneckBlock(64, 128, BN=BN)
        self.residual4 = module.BottleneckBlock(128, 128, BN=BN)

        self.residual5 = module.BottleneckBlock(128, 128, BN=BN)
        self.residual6 = module.BottleneckBlock(128, 256, BN=BN)
        self.residual7 = module.BottleneckBlock(256, 256, BN=BN)
        self.residual8 = module.BottleneckBlock(256, 256, BN=BN)

        self.fc = module.FullConnection(6400, 1, layer_num=2, dropout=dropout)

    def forward(self, x):
        # print(x)
        input_size = x.size(0)  # batch数
        # 在1维度（从左往右数，split的第二个参数1）以1（split的第二个参数）为大小，拆分x
        x1, x2 = x.split(1, 1)  # x1,x2:[batch, 1, 240, 240]

        x1 = self.conv_big(x1)
        x1 = F.relu(x1)
        x1 = self.residual0(x1)
        x1 = F.relu(x1)  # x1:[batch, 128, 240, 240]

        x2 = self.conv_big(x2)
        x2 = F.relu(x2)
        x2 = self.residual0(x2)
        x2 = F.relu(x2)  # x2:[batch, 128, 240, 240]
        # 在1维度（从左往右数），合并x1、x2
        x_merge = torch.cat([x1, x2], dim=1)  # x_merge:[batch, 256, 240, 240]

        x_merge = self.conv1(x_merge)
        x_merge = F.relu(x_merge)
        x_merge = F.max_pool2d(x_merge, 2, 2)

        x_merge = self.residual1(x_merge)
        x_merge = F.relu(x_merge)

        x_merge = self.residual2(x_merge)
        x_merge = F.relu(x_merge)
        x_merge = F.max_pool2d(x_merge, 3, 3)

        x_merge = self.residual3(x_merge)
        x_merge = F.relu(x_merge)

        x_merge = self.residual4(x_merge)
        x_merge = F.relu(x_merge)
        x_merge = F.max_pool2d(x_merge, 2, 2)

        x_merge = self.residual5(x_merge)
        x_merge = F.relu(x_merge)

        x_merge = self.residual6(x_merge)
        x_merge = F.relu(x_merge)

        x_merge = self.residual7(x_merge)
        x_merge = F.relu(x_merge)

        x_merge = self.residual8(x_merge)
        x_merge = F.relu(x_merge)
        x_merge = F.avg_pool2d(x_merge, 4, 4)

        x_merge = x_merge.view(input_size, -1)

        return self.fc(x_merge)


class BottleneckCNN(nn.Module):
    def __init__(self, BN=True, dropout=0):
        super().__init__()  # Net -> CNN
        # Conv2d: input_channels, output_channels, kernel_size
        # param_num: input_channels * output_channels * kernel_size * kernel_size
        # padding是在外围补充的圈数，padding=1就补1圈，为了和原图尺寸保持一致（别忘了，补1圈，边长+2！）
        # 如果不加padding的话，卷积后图像尺寸变化为：out_size = in_size - kernel_size + 1
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.residual1 = module.BottleneckBlock(64, 128, BN=BN)
        self.residual2 = module.BottleneckBlock(128, 64, BN=BN)
        self.residual3 = module.BottleneckBlock(64, 64, BN=BN)

        self.fc = module.FullConnection(1600, 1, layer_num=2, dropout=dropout)

    def forward(self, x):
        input_size = x.size(0)
        # print(input_size)
        # print(x.shape)
        # in: batch*1*240*240, out: batch*64*238*238
        x = self.conv1(x)
        # print(x.shape)
        # out: batch*64*240*240
        x = F.relu(x)
        # in: batch*64*240*240, out: batch*64*120*120
        x = F.max_pool2d(x, 2, 2)
        # print(x.shape)

        # in: batch*64*120*120, out: batch*128*120*120
        x = self.residual1(x)
        # print(x.shape)
        x = F.relu(x)
        # in: batch*128*120*120, out: batch*128*40*40
        x = F.max_pool2d(x, 3, 3)
        # print(x.shape)

        # in: batch*128*40*40, out: batch*64*40*40
        x = self.residual2(x)
        # print(x.shape)
        x = F.relu(x)
        # in: batch*64*40*40, out: batch*64*20*20
        x = F.max_pool2d(x, 2, 2)
        # print(x.shape)

        # in: batch*64*20*20, out: batch*64*20*20
        x = self.residual3(x)
        # print(x.shape)
        x = F.relu(x)
        # in: batch*64*20*20, out: batch*64*5*5
        # 一般最后一层池化用平均，其他用max
        # 因为前面层包含很多无用信息，用max可筛除之
        # 而到最后已经剔除过了很多无用信息，剩下的都是金贵的信息
        # 用平均可以更好的利用这些剩下的信息
        x = F.avg_pool2d(x, 4, 4)
        # print(x.shape)

        # batch*64*5*5
        x = x.view(input_size, -1)
        return self.fc(x)

