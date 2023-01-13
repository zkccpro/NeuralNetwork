import torchvision.models as models
from torch import nn
import torch
from model import net
from conf import globalParam


class DuelingDQN(nn.Module):
    def __init__(self, action_num, dropout=False):
        super().__init__()
        self.backbone = models.resnet18().to(globalParam.device)
        # 动作支路
        self.a_side = net.FullConnection(1000, action_num, layer_num=2, dropout=dropout)
        # 竞争支路
        self.v_side = net.FullConnection(1000, 1, layer_num=2, dropout=dropout)

    def forward(self, stat):
        feature_map = self.backbone(stat)
        a = self.a_side(feature_map)
        v = self.v_side(feature_map)
        return a + v


class DoubleMinDQN(nn.Module):
    def __init__(self, action_num, dropout=False):
        super().__init__()
        self.q1 = DuelingDQN(action_num, dropout=dropout)
        self.q2 = DuelingDQN(action_num, dropout=dropout)

    def forward(self, stat):
        x1 = self.q1(stat)
        x2 = self.q2(stat)
        return torch.min(x1, x2), x1, x2


class ValueNet(nn.Module):
    def __init__(self, dropout=False):
        super().__init__()
        self.backbone = models.resnet18().to(globalParam.device)
        self.fc = net.FullConnection(1000, 1, layer_num=2, dropout=dropout)

    def forward(self, stat):
        x = self.backbone(stat)
        x = self.fc(x)
        return x

