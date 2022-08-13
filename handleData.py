import globalParam
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


# 从磁盘读数据的接口，包括一个读数据的方法和一个数据路径
class DataReaderInterface:
    def __init__(self, path):
        self.path = path
        pic_num = 0
        for _, _, filenames in os.walk(path):
            pic_num = len(filenames)
        self.path_pic_num = pic_num

    # 读数据方法
    # total_num：训练+测试的所有数据
    # train_num：参与训练的所有数据
    # default=True时：采用路径中所有图片进行训练+测试，测试集规模为500
    # min_stat：最小显示进度的分度，为0时不显示进度，默认为20，即 每读取20分之一的图片显示一次进度
    # 返回值：{pic_list, test_list, label, test_label}四元组
    def read(self, total_num=0, train_num=0, default=False, min_stat=20):
        pass


# 普通的DataReader，读取一组图像作为单通道的输入，用于单个图片作为输入的网络
class DataReader(DataReaderInterface):
    def __init__(self, path):
        super(DataReader, self).__init__(path)

    def read(self, total_num=0, train_num=0, default=False, min_stat=20):
        if default:
            total_num = self.path_pic_num
            train_num = total_num - 500
        if total_num <= train_num:
            raise ValueError("total_num <= train_num!")
        pic_list = []
        test_list = []
        label = []
        test_label = []

        # 读图片、标签
        i = 0
        stat_every = 0
        if min_stat != 0:
            stat_every = int(total_num / min_stat)
        print("starting reading data, total num =", total_num, ", train_num =", train_num)
        for pic in os.listdir(self.path):
            if i >= total_num:
                break
            if min_stat != 0 and i % stat_every == 1:  # 显示进度
                stat = int((i / total_num) * 100)
                print('loading data completed:', stat, '%')

            img = Image.open(os.path.join(self.path, pic)).convert('L')
            tag = float(pic[len(pic) - 8: len(pic) - 4])
            # 这里细节需要十分注意！最外层一定要加一个[]，否则后面训练会崩，为啥呢？
            # 你前面是把一个单通道图像转换成np array，就导致train_x[i]只有二维！这样就会导致batch>1的时候，
            # dataloader会把多个二维的图像叠成一个batch【通道】的图像，就造成本来应该是一个四维tensor变成一个三维tensor
            # 你设置的batch数变成了通道数，正确做法就是在这里给train_x[i]添加一维，变成三维的单通道np array
            if i <= train_num:  # 训练
                pic_list.append([img])
                label.append(tag)
            else:  # 测试
                test_list.append([img])
                test_label.append(tag)
            i += 1

        return pic_list, test_list, label, test_label


# 分别读取组两组图像组成双通道的输入，用于两个输入图片的网络（两路输入网络）
class DataReaderDouble(DataReaderInterface):
    def __init__(self, path, path_2):
        super(DataReaderDouble, self).__init__(path)
        self.path_2 = path_2

    def read(self, total_num=0, train_num=0, default=False, min_stat=20):
        if default:
            total_num = self.path_pic_num
            train_num = total_num - 500
        if total_num <= train_num:
            raise ValueError("total_num <= train_num!")
        pic_list = []
        test_list = []
        label = []
        test_label = []

        # 读图片、标签
        i = 0
        stat_every = 0
        if min_stat != 0:
            stat_every = int(total_num / min_stat)
        print("starting reading data, total num =", total_num, ", train_num =", train_num)
        for pic in os.listdir(self.path):
            if i >= total_num:
                break
            if min_stat != 0 and i % stat_every == 1:  # 显示进度
                stat = int((i / total_num) * 100)
                print('loading data completed:', stat, '%')
            img = Image.open(os.path.join(self.path, pic)).convert('L')
            img_2 = Image.open(os.path.join(self.path_2, pic)).convert('L')
            tag = float(pic[len(pic) - 8: len(pic) - 4])
            if i <= train_num:  # 训练
                pic_list.append([img, img_2])
                label.append(tag)
            else:  # 测试
                test_list.append([img, img_2])
                test_label.append(tag)
            i += 1
        return pic_list, test_list, label, test_label


# 集成调整数据格式，供DataLoader做训练前的最终处理
# 注意，训练集大小必须比测试集大，否则会出问题
# data_type规定有2钟：
# 1. 'img'（默认）
# 2. 'tensor'
# 不允许有list类型的数据！（既不是tensor又不是PIL img是不允许的）
class CustomedDataSet(Dataset):
    def __init__(self, train=True, train_x=None, train_y=None, test_x=None, test_y=None,
                 transform=None, target_transform=None,
                 data_transform=None, target_data_transform=None, data_type='img'):
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.data_transform = data_transform
        self.target_data_transform = target_data_transform
        self.data_type = data_type
        # 图像变换
        # 能处理三维和四维的输入，而且第四维必须是由list组织的；1,2维处理不了
        if self.transform is not None:
            if self.train:
                for channel in train_x:
                    for i in range(len(channel)):
                        channel[i] = self.transform(channel[i])
            else:
                for channel in test_x:
                    for i in range(len(channel)):
                        channel[i] = self.transform(channel[i])
        # 标签变换
        # 能处理1,2,3,4维的标签，但是四维的时候，第四维不能是list组织的
        if self.target_transform is not None:
            if self.train:
                for i in range(len(train_y)):
                    train_y[i] = self.target_transform(train_y[i])
            else:
                for i in range(len(test_y)):
                    test_y[i] = self.target_transform(test_y[i])
        # 图像数据类型转换（数据和标签都转成np）
        if self.train:
            if data_transform:
                self.dataset = data_transform(train_x)
            else:
                self.dataset = train_x
            if target_data_transform:
                self.labels = target_data_transform(train_y)
            else:
                self.labels = train_y
        else:
            if data_transform:
                self.dataset = data_transform(test_x)
            else:
                self.dataset = test_x
            if target_data_transform:
                self.labels = target_data_transform(test_y)
            else:
                self.labels = test_y

        if self.data_type == 'img':
            if self.train:
                for channel in train_x:
                    for i in range(len(channel)):
                        channel[i] = np.asarray(channel[i])
                        # 归一化
                        section = np.max(channel[i]) - np.min(channel[i])
                        if section != 0:
                            channel[i] = (channel[i] - np.min(channel[i])) / section
                train_x = np.array(train_x)
                # print(train_x.shape)
                for i in range(len(train_y)):
                    train_y[i] = [train_y[i]]
                train_y = np.array(train_y)
            else:
                for channel in test_x:
                    for i in range(len(channel)):
                        channel[i] = np.asarray(channel[i])
                        # 归一化
                        section = np.max(channel[i]) - np.min(channel[i])
                        if section != 0:
                            channel[i] = (channel[i] - np.min(channel[i])) / section
                test_x = np.array(test_x)
                for i in range(len(test_y)):
                    test_y[i] = [test_y[i]]
                test_y = np.array(test_y)

        # 赋给成员变量，如果数据类型不是tensor，就转成tensor再赋值；是tensor就直接扔进去
        if self.data_type != 'tensor':
            if self.train:
                # 这里将造成内存暴涨，容易OOM，建议训练前把能关掉的进程通通关掉，训练前内存占用20%-30%才行，训练集样本极限也就1w了
                self.dataset = torch.from_numpy(train_x).float()
                self.labels = torch.from_numpy(train_y).float()  # for mse loss: float, nll loss: long
            else:
                self.dataset = torch.from_numpy(test_x).float()
                self.labels = torch.from_numpy(test_y).float()  # for mse loss: float, nll loss: long
                # print(self.labels)
        else:
            if self.train:
                self.dataset = train_x.float()
                self.labels = train_y.float()  # for mse loss: float, nll loss: long
            else:
                self.dataset = test_x.float()
                self.labels = test_y.float()  # for mse loss: float,nll loss: long

    def __getitem__(self, index):
        if self.train:
            return self.dataset[index].to(globalParam.device), self.labels[index].to(globalParam.device)
        else:
            return self.dataset[index].to(globalParam.device), self.labels[index].to(globalParam.device)

    def __len__(self):
        return self.dataset.shape[0]


