import model
import torch
import PIL
import numpy as np

# cuda or cpu
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
print(device)
print(torch.cuda.get_device_properties(0))  # 显卡详细信息，显卡型号，显存等

# 网络
nn_network = model.NN().to(device)
residualcnn_network = model.ResidualCNN().to(device)
bottleneckcnn_network = model.BottleneckCNN().to(device)
bottleneckcnn_doubleinput_network = model.BottleneckCNNDoubleInput().to(device)
secnn_doubleinput_network = model.SECNNDoubleInput().to(device)
unet_network = model.Unet().to(device)

# 优化器
# lr:学习率，不宜过大, momentum:优化器冲量
# weight_decay: 权重衰减系数（加上这个就相当于对权重进行L2范数约束，这个参数就相当于λ）
nn_optimizer = torch.optim.SGD(nn_network.parameters(), lr=0.2)
residualcnn_optimizer = torch.optim.SGD(residualcnn_network.parameters(), lr=0.00001, momentum=0.9, weight_decay=1e-2)
bottleneckcnn_optimizer = torch.optim.SGD(bottleneckcnn_network.parameters(), lr=0.00002, momentum=0.9, weight_decay=1e-2)
bottleneckcnn_doubleinput_optimizer = torch.optim.SGD(bottleneckcnn_doubleinput_network.parameters(), lr=0.00005, momentum=0.9, weight_decay=1e-2)
secnn_doubleinput_optimizer = torch.optim.SGD(secnn_doubleinput_network.parameters(), lr=0.00005, momentum=0.9, weight_decay=1e-2)
unet_optimizer = torch.optim.SGD(unet_network.parameters(), lr=0.00005, momentum=0.9, weight_decay=1e-2)


# 图像变换函数（预处理）
def pic_transfer(src):
    dst = src.resize((240, 240), PIL.Image.ANTIALIAS)
    return dst


# 标签变换函数（预处理）
def label_transfer(label):
    return label / 2  # 标签做归一化


# 图像数据转换，用于把三维和四维[PIL image]数据（data/label均可）类型转换成tensor类型
# 三维图像中的第三维必须是由python list组织的
def pic_data_transfer(data):
    for channel in data:
        for i in range(len(channel)):
            channel[i] = np.asarray(channel[i])
            # 归一化
            section = np.max(channel[i]) - np.min(channel[i])
            if section != 0:
                channel[i] = (channel[i] - np.min(channel[i])) / section
    return torch.from_numpy(np.array(data)).float()  # for mse loss: float,nll loss: long


# 单通道图像（没有手动升维的图像）数据转换，用于把三维[PIL image]数据（data/label均可）类型转换成tensor类型
# 一般用于label图像的数据转换（因为label图像是不做手动升维的）
def label_pic_data_transfer(data):
    for i in range(len(data)):
        data[i] = np.expand_dims(np.asarray(data[i]), axis=0)
        # 归一化
        section = np.max(data[i]) - np.min(data[i])
        if section != 0:
            data[i] = (data[i] - np.min(data[i])) / section
    ret = torch.from_numpy(np.array(data)).float()
    return ret  # for mse loss: float,nll loss: long


# 一维数据转换，用于把一维的数据（data/label均可）类型转换为tensor类型
def num_data_transfer(data):
    for i in range(len(data)):
        data[i] = [data[i]]
    return torch.from_numpy(np.array(data)).float()  # for mse loss: float,nll loss: long

