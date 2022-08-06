from torch import nn
import torch.nn.functional as F
import module
import torch


# 根据博客中使用的手写字识别网络稍微修改
# input: tensor[batch, 1, 240, 240], 单通道，240*240图像
# output:tensor[1]，评分；范围：[-1,1]
# 参数总数：1.3亿
# 卷积层参数：2058
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()  # Net -> CNN
        # Conv2d: input_channels, output_channels, kernel_size
        # param_num: input_channels * output_channels * kernel_size * kernel_size
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.conv2 = nn.Conv2d(10, 20, 3)

        self.fc1 = nn.Linear(20*58*58, 2000)
        self.fc2 = nn.Linear(2000, 1)

    def forward(self, x):
        input_size = x.size(0)
        # print(x.shape)
        # in: batch*1*240*240, out: batch*10*236*236(240-5+1)
        x = self.conv1(x)
        # out: batch*10*236*236
        x = F.relu(x)
        # in: batch*10*236*236, out: batch*10*118*118
        x = F.max_pool2d(x,2,2)
        # in: batch*10*118*118, out: batch*20*116*116 (118-3+1)
        x = self.conv2(x)
        x = F.relu(x)
        # in: batch*20*116*116, out: batch*20*58*58
        x = F.max_pool2d(x, 2, 2)

        # batch*20*116*116
        x = x.view(input_size,-1)

        # in: batch*20*116*116  out:batch*2000
        x = self.fc1(x)
        x = F.relu(x)

        # in:batch*2000 out:batch*1
        x = self.fc2(x)
        return x


class ResidualCNN(nn.Module):
    def __init__(self):
        super(ResidualCNN, self).__init__()  # Net -> CNN
        # Conv2d: input_channels, output_channels, kernel_size
        # param_num: input_channels * output_channels * kernel_size * kernel_size
        # padding是在外围补充的圈数，padding=1就补1圈，为了和原图尺寸保持一致（别忘了，补1圈，边长+2！）
        # 如果不加padding的话，卷积后图像尺寸变化为：out_size = in_size - kernel_size + 1
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.residual1 = module.ResidualBlock(64, 128)
        self.residual2 = module.ResidualBlock(128, 64)
        self.residual3 = module.ResidualBlock(64, 64)

        self.fc1 = nn.Linear(1600, 200)
        self.fc2 = nn.Linear(200, 1)

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
        x = self.fc1(x)
        x = F.relu(x)

        # in:batch*200 out:batch*1
        x = self.fc2(x)
        return x


class BottleneckCNNDoubleInput(nn.Module):
    def __init__(self):
        super(BottleneckCNNDoubleInput, self).__init__()  # Net -> CNN
        # Conv2d: input_channels, output_channels, kernel_size
        # param_num: input_channels * output_channels * kernel_size * kernel_size
        # padding是在外围补充的圈数，padding=1就补1圈，为了和原图尺寸保持一致（别忘了，补1圈，边长+2！）
        # 如果不加padding的话，卷积后图像尺寸变化为：out_size = in_size - kernel_size + 1
        self.conv_big = nn.Conv2d(1, 32, kernel_size=7, padding=3)
        self.residual0 = module.BottleneckBlock(32, 32, 8)

        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.residual1 = module.BottleneckBlock(64, 64, 8)
        self.residual2 = module.BottleneckBlock(64, 64, 8)
        self.residual3 = module.BottleneckBlock(64, 64, 8)

        self.fc1 = nn.Linear(1600, 100)
        self.fc2 = nn.Linear(100, 1)

    def forward(self, x):
        input_size = x.size(0)  # batch数
        # print(x.shape)
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
        # print(x_merge.shape)

        # in: batch*256*240*240, out: batch*256*240*240
        x_merge = self.conv1(x_merge)
        # print(x_merge.shape)
        # out: batch*256*240*240
        x_merge = F.relu(x_merge)
        # in: batch*256*240*240, out: batch*256*120*120
        x_merge = F.max_pool2d(x_merge, 2, 2)
        # print(x_merge.shape)

        # in: batch*256*120*120, out: batch*256*120*120
        x_merge = self.residual1(x_merge)
        # print(x_merge.shape)
        x_merge = F.relu(x_merge)
        # in: batch*256*120*120, out: batch*256*40*40
        x_merge = F.max_pool2d(x_merge, 3, 3)
        # print(x_merge.shape)

        # in: batch*256*120*120, out: batch*256*120*120
        x_merge = self.residual2(x_merge)
        # print(x_merge.shape)
        x_merge = F.relu(x_merge)
        # in: batch*256*120*120, out: batch*256*40*40
        x_merge = F.max_pool2d(x_merge, 2, 2)
        # print(x_merge.shape)

        # in: batch*512*20*20, out: batch*512*20*20
        x_merge = self.residual3(x_merge)
        # print(x_merge.shape)
        x_merge = F.relu(x_merge)
        # 一般最后一层池化用平均，其他用max
        # 因为前面层包含很多无用信息，用max可筛除之
        # 而到最后已经剔除过了很多无用信息，剩下的都是金贵的信息
        # 用平均可以更好的利用这些剩下的信息
        # in: batch*512*20*20, out: batch*512*5*5
        x_merge = F.avg_pool2d(x_merge, 4, 4)
        # print(x_merge.shape)

        # batch*512*5*5
        x_merge = x_merge.view(input_size, -1)
        # print(x_merge.shape)
        # in: batch*512*5*5=batch*12800  out:batch*1000
        x_merge = self.fc1(x_merge)
        x_merge = F.relu(x_merge)

        # in:batch*1000 out:batch*1
        x_merge = self.fc2(x_merge)
        return x_merge


class BottleneckCNN(nn.Module):
    def __init__(self):
        super(BottleneckCNN, self).__init__()  # Net -> CNN
        # Conv2d: input_channels, output_channels, kernel_size
        # param_num: input_channels * output_channels * kernel_size * kernel_size
        # padding是在外围补充的圈数，padding=1就补1圈，为了和原图尺寸保持一致（别忘了，补1圈，边长+2！）
        # 如果不加padding的话，卷积后图像尺寸变化为：out_size = in_size - kernel_size + 1
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.residual1 = module.BottleneckBlock(64, 128, 32)
        self.residual2 = module.BottleneckBlock(128, 64, 32)
        self.residual3 = module.BottleneckBlock(64, 64, 32)

        self.fc1 = nn.Linear(1600, 200)
        self.fc2 = nn.Linear(200, 1)

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
        # print(x.shape)
        # in: batch*64*5*5=batch*1600  out:batch*200
        x = self.fc1(x)
        x = F.relu(x)

        # in:batch*200 out:batch*1
        x = self.fc2(x)
        return x

# 秦铁毕业论文中网络的卷积部分
# input: tensor[batch, 1, 240, 240], 单通道，240*240图像
# output:tensor[1]，评分；范围：[-1,1]
# 参数总数：50.5w
# 卷积层参数：184929
class BigCNN(nn.Module):
    def __init__(self):
        super(BigCNN, self).__init__()  # Net -> CNN
        # Conv2d: input_channels, output_channels, kernel_size
        # param_num: input_channels * output_channels * kernel_size * kernel_size
        # padding是在外围补充的圈数，padding=1就补1圈，为了和原图尺寸保持一致（别忘了，补1圈，边长+2！）
        # 如果不加padding的话，卷积后图像尺寸变化为：out_size = in_size - kernel_size + 1
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(1600, 200)
        self.fc2 = nn.Linear(200, 1)

    def forward(self, x):
        input_size = x.size(0)
        print(x.shape)
        # in: batch*1*240*240, out: batch*64*238*238
        x = self.conv1(x)
        print(x.shape)
        # out: batch*64*240*240
        x = F.relu(x)
        # in: batch*64*240*240, out: batch*64*120*120
        x = F.max_pool2d(x, 2, 2)
        print(x.shape)

        # in: batch*64*120*120, out: batch*128*120*120
        x = self.conv2(x)
        print(x.shape)
        x = F.relu(x)
        # in: batch*128*120*120, out: batch*128*40*40
        x = F.max_pool2d(x, 3, 3)
        print(x.shape)

        # in: batch*128*40*40, out: batch*64*40*40
        x = self.conv3(x)
        print(x.shape)
        x = F.relu(x)
        # in: batch*64*40*40, out: batch*64*20*20
        x = F.max_pool2d(x, 2, 2)
        print(x.shape)

        # in: batch*64*20*20, out: batch*64*20*20
        x = self.conv4(x)
        print(x.shape)
        x = F.relu(x)
        # in: batch*64*20*20, out: batch*64*5*5
        x = F.avg_pool2d(x, 4, 4)
        print(x.shape)

        # batch*64*5*5
        x = x.view(input_size, -1)
        print(x.shape)
        # in: batch*64*5*5=batch*1600  out:batch*200
        x = self.fc1(x)
        x = F.relu(x)

        # in:batch*200 out:batch*1
        x = self.fc2(x)
        return x


class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()  # Net -> CNN

        self.fc1 = nn.Linear(1, 10)
        self.fc2 = nn.Linear(10, 20)
        self.fc3 = nn.Linear(20, 1)

    def forward(self, x):
        print(x.shape)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x

