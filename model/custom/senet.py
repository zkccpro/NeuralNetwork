from torch import nn
import torch.nn.functional as F
from model import module
from model import net
import torch


class SECNNDoubleInput(nn.Module):
    def __init__(self, BN=True, dropout=0):
        super().__init__()  # Net -> CNN
        # Conv2d: input_channels, output_channels, kernel_size
        # param_num: input_channels * output_channels * kernel_size * kernel_size
        # padding是在外围补充的圈数，padding=1就补1圈，为了和原图尺寸保持一致（别忘了，补1圈，边长+2！）
        # 如果不加padding的话，卷积后图像尺寸变化为：out_size = in_size - kernel_size + 1
        self.conv_big = nn.Conv2d(1, 32, kernel_size=7, padding=3)
        self.residual0 = module.SEBlock(32, 32, BN=BN)

        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.residual1 = module.SEBlock(64, 64, BN=BN)
        self.residual2 = module.SEBlock(64, 64, BN=BN)

        self.residual3 = module.SEBlock(64, 128, BN=BN)
        self.residual4 = module.SEBlock(128, 128, BN=BN)

        self.residual5 = module.SEBlock(128, 128, BN=BN)
        self.residual6 = module.SEBlock(128, 256, BN=BN)
        self.residual7 = module.SEBlock(256, 256, BN=BN)
        self.residual8 = module.SEBlock(256, 256, BN=BN)

        self.fc = net.FullConnection(6400, 1, layer_num=2, dropout=dropout)

    def forward(self, x):
        # print(x)
        input_size = x.size(0)  # batch数
        # 在1维度（从左往右数，split的第二个参数1）以1（split的第二个参数）为大小，拆分x
        x1, x2 = x.split(1, 1)  # x1,x2:[batch, 1, 240, 240]

        x1 = self.conv_big(x1)
        x1 = F.relu(x1)
        x1 = self.residual0(x1)  # x1:[batch, 128, 240, 240]

        x2 = self.conv_big(x2)
        x2 = F.relu(x2)
        x2 = self.residual0(x2)  # x2:[batch, 128, 240, 240]
        # 在1维度（从左往右数），合并x1、x2
        x_merge = torch.cat([x1, x2], dim=1)  # x_merge:[batch, 256, 240, 240]

        x_merge = self.conv1(x_merge)
        x_merge = F.relu(x_merge)
        x_merge = F.max_pool2d(x_merge, 2, 2)

        x_merge = self.residual1(x_merge)

        x_merge = self.residual2(x_merge)
        x_merge = F.max_pool2d(x_merge, 3, 3)

        x_merge = self.residual3(x_merge)

        x_merge = self.residual4(x_merge)
        x_merge = F.max_pool2d(x_merge, 2, 2)

        x_merge = self.residual5(x_merge)

        x_merge = self.residual6(x_merge)

        x_merge = self.residual7(x_merge)

        x_merge = self.residual8(x_merge)
        x_merge = F.avg_pool2d(x_merge, 4, 4)

        x_merge = x_merge.view(input_size, -1)

        return self.fc(x_merge)
        # x_merge = self.fc1(x_merge)
        # x_merge = F.relu(x_merge)

        # x_merge = self.fc2(x_merge)  # 注意，回归问题输出层不能再加relu，否则输出全0！
        # return x_merge


class BigSECNNDoubleInput(nn.Module):
    def __init__(self, BN=True, dropout=0):
        super().__init__()  # Net -> CNN
        # Conv2d: input_channels, output_channels, kernel_size
        # param_num: input_channels * output_channels * kernel_size * kernel_size
        # padding是在外围补充的圈数，padding=1就补1圈，为了和原图尺寸保持一致（别忘了，补1圈，边长+2！）
        # 如果不加padding的话，卷积后图像尺寸变化为：out_size = in_size - kernel_size + 1
        self.conv_big = nn.Conv2d(1, 32, kernel_size=7, padding=3)
        self.residual0 = module.SEBlock(32, 32, BN=BN)

        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.residual1 = module.SEBlock(64, 64, BN=BN)
        self.residual1_1 = module.SEBlock(64, 64, BN=BN)
        self.residual1_2 = module.SEBlock(64, 64, BN=BN)
        self.residual1_3 = module.SEBlock(64, 64, BN=BN)
        self.residual1_4 = module.SEBlock(64, 64, BN=BN)

        self.residual2 = module.SEBlock(64, 64, BN=BN)
        self.residual2_1 = module.SEBlock(64, 64, BN=BN)
        self.residual2_2 = module.SEBlock(64, 64, BN=BN)
        self.residual2_3 = module.SEBlock(64, 64, BN=BN)
        self.residual2_4 = module.SEBlock(64, 64, BN=BN)

        self.residual3 = module.SEBlock(64, 128, BN=BN)
        self.residual3_1 = module.SEBlock(128, 128, BN=BN)
        self.residual3_2 = module.SEBlock(128, 128, BN=BN)
        self.residual3_3 = module.SEBlock(128, 128, BN=BN)
        self.residual3_4 = module.SEBlock(128, 128, BN=BN)

        self.residual4 = module.SEBlock(128, 128, BN=BN)
        self.residual4_1 = module.SEBlock(128, 128, BN=BN)
        self.residual4_2 = module.SEBlock(128, 128, BN=BN)
        self.residual4_3 = module.SEBlock(128, 128, BN=BN)
        self.residual4_4 = module.SEBlock(128, 128, BN=BN)

        self.residual5 = module.SEBlock(128, 128, BN=BN)
        self.residual5_1 = module.SEBlock(128, 128, BN=BN)
        self.residual5_2 = module.SEBlock(128, 128, BN=BN)
        self.residual5_3 = module.SEBlock(128, 128, BN=BN)
        self.residual5_4 = module.SEBlock(128, 128, BN=BN)

        self.residual6 = module.SEBlock(128, 256, BN=BN)
        self.residual6_1 = module.SEBlock(256, 256, BN=BN)
        self.residual6_2 = module.SEBlock(256, 256, BN=BN)
        self.residual6_3 = module.SEBlock(256, 256, BN=BN)
        self.residual6_4 = module.SEBlock(256, 256, BN=BN)

        self.residual7 = module.SEBlock(256, 256, BN=BN)
        self.residual7_1 = module.SEBlock(256, 256, BN=BN)
        self.residual7_2 = module.SEBlock(256, 256, BN=BN)
        self.residual7_3 = module.SEBlock(256, 256, BN=BN)
        self.residual7_4 = module.SEBlock(256, 256, BN=BN)

        self.residual8 = module.SEBlock(256, 256, BN=BN)
        self.residual8_1 = module.SEBlock(256, 256, BN=BN)
        self.residual8_2 = module.SEBlock(256, 256, BN=BN)
        self.residual8_3 = module.SEBlock(256, 256, BN=BN)
        self.residual8_4 = module.SEBlock(256, 256, BN=BN)

        self.fc = net.FullConnection(6400, 1, layer_num=2, dropout=dropout)

    def forward(self, x):
        # print(x)
        input_size = x.size(0)  # batch数
        # 在1维度（从左往右数，split的第二个参数1）以1（split的第二个参数）为大小，拆分x
        x1, x2 = x.split(1, 1)  # x1,x2:[batch, 1, 240, 240]

        x1 = self.conv_big(x1)
        x1 = F.relu(x1)
        x1 = self.residual0(x1)  # x1:[batch, 128, 240, 240]

        x2 = self.conv_big(x2)
        x2 = F.relu(x2)
        x2 = self.residual0(x2)  # x2:[batch, 128, 240, 240]
        # 在1维度（从左往右数），合并x1、x2
        x_merge = torch.cat([x1, x2], dim=1)  # x_merge:[batch, 256, 240, 240]

        x_merge = self.conv1(x_merge)
        x_merge = F.relu(x_merge)
        x_merge = F.max_pool2d(x_merge, 2, 2)

        x_merge = self.residual1(x_merge)
        x_merge = self.residual1_1(x_merge)
        x_merge = self.residual1_2(x_merge)
        x_merge = self.residual1_3(x_merge)
        x_merge = self.residual1_4(x_merge)

        x_merge = self.residual2(x_merge)
        x_merge = self.residual2_1(x_merge)
        x_merge = self.residual2_2(x_merge)
        x_merge = self.residual2_3(x_merge)
        x_merge = self.residual2_4(x_merge)
        x_merge = F.max_pool2d(x_merge, 3, 3)

        x_merge = self.residual3(x_merge)
        x_merge = self.residual3_1(x_merge)
        x_merge = self.residual3_2(x_merge)
        x_merge = self.residual3_3(x_merge)
        x_merge = self.residual3_4(x_merge)

        x_merge = self.residual4(x_merge)
        x_merge = self.residual4_1(x_merge)
        x_merge = self.residual4_2(x_merge)
        x_merge = self.residual4_3(x_merge)
        x_merge = self.residual4_4(x_merge)
        x_merge = F.max_pool2d(x_merge, 2, 2)

        x_merge = self.residual5(x_merge)
        x_merge = self.residual5_1(x_merge)
        x_merge = self.residual5_2(x_merge)
        x_merge = self.residual5_3(x_merge)
        x_merge = self.residual5_4(x_merge)

        x_merge = self.residual6(x_merge)
        x_merge = self.residual6_1(x_merge)
        x_merge = self.residual6_2(x_merge)
        x_merge = self.residual6_3(x_merge)
        x_merge = self.residual6_4(x_merge)

        x_merge = self.residual7(x_merge)
        x_merge = self.residual7_1(x_merge)
        x_merge = self.residual7_2(x_merge)
        x_merge = self.residual7_3(x_merge)
        x_merge = self.residual7_4(x_merge)

        x_merge = self.residual8(x_merge)
        x_merge = self.residual8_1(x_merge)
        x_merge = self.residual8_2(x_merge)
        x_merge = self.residual8_3(x_merge)
        x_merge = self.residual8_4(x_merge)
        x_merge = F.avg_pool2d(x_merge, 4, 4)

        x_merge = x_merge.view(input_size, -1)

        return self.fc(x_merge)
        # x_merge = self.fc1(x_merge)
        # x_merge = F.relu(x_merge)

        # x_merge = self.fc2(x_merge)  # 注意，回归问题输出层不能再加relu，否则输出全0！
        # return x_merge
