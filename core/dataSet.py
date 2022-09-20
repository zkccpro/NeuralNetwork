from conf import globalParam
from torch.utils.data import Dataset


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
        self.data_type = data_type
        # 图像变换
        # 能处理三维和四维的输入，而且第四维必须是由list组织的；1,2维处理不了
        if transform:
            if self.train:
                for channel in train_x:
                    for i in range(len(channel)):
                        channel[i] = transform(channel[i])
            else:
                for channel in test_x:
                    for i in range(len(channel)):
                        channel[i] = transform(channel[i])
        # 标签变换
        # 能处理1,2,3,4维的标签，但是四维的时候，第四维不能是list组织的
        if target_transform:
            if self.train:
                for i in range(len(train_y)):
                    train_y[i] = target_transform(train_y[i])
            else:
                for i in range(len(test_y)):
                    test_y[i] = target_transform(test_y[i])
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

    def __getitem__(self, index):
        if self.train:
            return self.dataset[index].to(globalParam.device), self.labels[index].to(globalParam.device)
        else:
            return self.dataset[index].to(globalParam.device), self.labels[index].to(globalParam.device)

    def __len__(self):
        return self.dataset.shape[0]

