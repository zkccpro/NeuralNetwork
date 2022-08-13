import handleData
import action
import globalParam
import draw
import torch
from torch.utils.data import DataLoader


def ut_nn():
    x = torch.unsqueeze(torch.linspace(-1, 1, 2), dim=1)
    y = x.pow(2) + 0.2 * torch.rand(x.size())
    test_x = torch.unsqueeze(torch.linspace(-1, 1, 1), dim=1)
    test_y = test_x.pow(2) + 0.2 * torch.rand(test_x.size())
    print(test_y)
    # prepare train data
    dataset = handleData.CustomedDataSet(train_x=x, train_y=y, transform=None, data_type='tensor')
    dataloader = DataLoader(dataset, batch_size=10, shuffle=False, sampler=None,
               batch_sampler=None, num_workers=0, collate_fn=None,
               pin_memory=False, drop_last=False, timeout=0,
               worker_init_fn=None)
    # prepare test data
    testset = handleData.CustomedDataSet(train=False, test_x=test_x, test_y=test_y, transform=None, data_type='tensor')
    testloader = DataLoader(testset, batch_size=1, shuffle=False, sampler=None,
               batch_sampler=None, num_workers=0, collate_fn=None,
               pin_memory=False, drop_last=False, timeout=0,
               worker_init_fn=None)
    # prepare model
    nn = globalParam.nn_network
    # train
    action.train(nn, dataloader, globalParam.nn_optimizer, epoch=100)
    # test
    action.test_regression(nn, testloader)


# 几乎是照搬博客上的代码，他这个训练方式相当于：batch=100，epoch=100
# batch=100：说明每一轮更新参数，用到了100个数据
# epoch=100：说明所有数据被反复利用了100次
def ut_nn_without_dataloader():
    x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
    y = x.pow(2) + 0.2 * torch.rand(x.size())
    # 定义优化器
    nn_net = globalParam.nn_network
    optimizer = torch.optim.SGD(nn_net.parameters(), lr=0.2)
    # 定义误差函数
    loss_fun = torch.nn.MSELoss()
    # 迭代训练
    loss_arr = []
    for t in range(1000):
        # 预测
        prediction = nn_net(x)
        # 计算误差
        loss = loss_fun(prediction, y)
        loss_arr.append(loss.data)
        # 梯度降为0
        optimizer.zero_grad()
        # 反向传递
        loss.backward()
        # 优化梯度
        optimizer.step()
    draw.draw_1d(loss_arr)