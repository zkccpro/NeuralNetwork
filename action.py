import torch
import draw
import globalParam
import utility


# epoch：训练代数，所有的数据将会被反复利用epoch次
def train(network, train_loader, test_loader, optimizer, epoch=1):
    loss_fun = torch.nn.MSELoss()
    epoch_loss = []  # 每个epoch的平均loss
    epoch_test_loss_mean = []  # 每个epoch在测试集上的平均loss
    status = 0
    for i in range(epoch):
        network.train()  # Batch Normalization 和 Dropout
        status += 1
        avg_loss = 0  # 每个epoch中所有iteration的平均loss
        print('training completed:', (status/epoch)*100, '%')
        j = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            output = network(data)
            loss = loss_fun(output, target)
            avg_loss += loss.data.to('cpu')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            j += 1

        avg_loss /= j
        epoch_loss.append(avg_loss)  # 按epoch打印loss
        print('avg loss =', avg_loss)  # 输出每个epoch中的平均loss

        test_loss, _ = test_regression(network, test_loader)
        test_loss_mean = utility.mean(test_loss)  # test_loader为空这里弹warning，test_loss暴nan这里弹warning
        epoch_test_loss_mean.append(test_loss_mean)
        print('test loss mean =', test_loss_mean)  # 输出该epoch训练后在测试集上的平均损失

    draw.draw_1d(epoch_loss, xlabel="epoch", ylabel="loss")
    draw.draw_1d(epoch_test_loss_mean, xlabel="epoch", ylabel="mean test loss")


def test_classify(network, testloader):
    for data, target in testloader:
        outputs = network(data)
        _, predicted = torch.max(outputs.data, 1)  # torch.max(data,1)：返回tensor中每行的最大值和索引
    print("prediction is: ", predicted)  # 这个输出要按你自己想要的调整的，这个输出是预测最大可能的数字


def test_regression(network, testloader, draw_or_not=False):
    predict_arr = []
    target_arr = []
    loss_arr = []
    loss_fun = torch.nn.MSELoss()
    for data, target in testloader:
        outputs = network(data)
        loss = loss_fun(outputs, target)
        loss_arr.append(loss.data.to('cpu'))
        predict_arr.append(outputs.data.to('cpu'))
        target_arr.append(target.data.to('cpu'))
    if draw_or_not:
        draw.draw_1d(loss_arr, xlabel='label', ylabel='loss')
        draw.drwa_2_data([target_arr, predict_arr], ['target', 'predict'], xlabel='label', ylabel='value')
    return loss_arr, predict_arr


# 只保存model的参数
# warning：保存前需确保路径有效！
def save_model(network, path):
    torch.save(network.state_dict(), path)  # 保存参数


# 加载model的参数到一个【同质】的空网络变量（同质是指网络结构一模一样，或者说和磁盘中的模型是同一个类的实例）
def model_playback(network, path):
    trained_net = network.to(globalParam.device)  # 创建一个网络变量
    trained_net.load_state_dict(torch.load(path))  # 导入参数
    # eval()不启用 Batch Normalization（BN） 和 Dropout，保证BN和dropout不发生变化，pytorch框架会自动把BN和Dropout固定住，不会取平均，
    # 而是用训练好的值，不然的话，一旦test的batch_size过小，很容易就会被BN层影响结果。
    trained_net.eval()

