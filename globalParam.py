import model
import torch
import PIL


# cuda or cpu
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'
print(device)
print(torch.cuda.get_device_properties(0))  # 显卡详细信息，显卡型号，显存等

# 网络
cnn_network = model.BigCNN().to(device)
nn_network = model.NN().to(device)
residualcnn_network = model.ResidualCNN().to(device)
bottleneckcnn_network = model.BottleneckCNN().to(device)
bottleneckcnn_doubleinput_network = model.BottleneckCNNDoubleInput().to(device)

# 优化器
# lr:学习率，不宜过大, momentum:优化器冲量
# weight_decay: 权重衰减系数（加上这个就相当于对权重进行L2范数约束，这个参数就相当于λ）
cnn_optimizer = torch.optim.SGD(cnn_network.parameters(), lr=0.00001, momentum=0.9, weight_decay=1e-2)
nn_optimizer = torch.optim.SGD(nn_network.parameters(), lr=0.2)
residualcnn_optimizer = torch.optim.SGD(residualcnn_network.parameters(), lr=0.00001, momentum=0.9, weight_decay=1e-2)
bottleneckcnn_optimizer = torch.optim.SGD(bottleneckcnn_network.parameters(), lr=0.00002, momentum=0.9, weight_decay=1e-2)
bottleneckcnn_doubleinput_optimizer = torch.optim.SGD(bottleneckcnn_doubleinput_network.parameters(), lr=0.00002, momentum=0.9, weight_decay=1e-2)


# 图像变换函数（预处理）
def pic_transfer(src):
    dst = src.resize((240, 240), PIL.Image.ANTIALIAS)
    return dst


# 标签变换函数（预处理）
def label_transfer(label):
    return label / 2  # 标签做归一化

