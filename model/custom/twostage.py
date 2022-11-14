from model.custom import resnet
from model.custom import gennet
from torch import nn
import torch


class TwostageNet(nn.Module):
    def __init__(self, BN=True, dropout=0):
        super().__init__()
        self.stage_one = gennet.Unet(BN=BN)
        self.stage_two = resnet.BottleneckCNNDoubleInput(BN=BN, dropout=dropout)

    def forward(self, x):
        x_diff = self.stage_one(x)  # 阶段1输出的偏差图像
        # x.shape:torch(batch, 1, 240, 240)
        # x_diff.shape:torch(batch, 1, 240, 240)
        # x_input2.shape:torch(batch, 2, 240, 240)
        x_input2 = torch.cat([x, x_diff], dim=1)
        return self.stage_two(x_input2)  # 阶段2输出最终曝光偏移
