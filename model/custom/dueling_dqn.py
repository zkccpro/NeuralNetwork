import torchvision.models as models
from torch import nn
import torch
from model import net
from conf import globalParam


class DuelingDQN(nn.Module):
    def __init__(self, dropout=False):
        super().__init__()
        self.backbone = models.resnet18().to(globalParam.device)
        # 动作支路
        self.a_side = net.FullConnection(1000, 7, layer_num=2, dropout=dropout)
        # 竞争支路
        self.v_side = net.FullConnection(1000, 1, layer_num=2, dropout=dropout)

    def forward(self, x):
        feature_map = self.backbone(x)
        a = self.a_side(feature_map)
        v = self.v_side(feature_map)
        return a + v
