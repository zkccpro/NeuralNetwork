import torch
from post_processing import draw
from post_processing import saver
from conf import globalParam
from parse import configParser as cp


# max_epoch：训练代数，所有的数据将会被反复利用epoch次
# iteration_show：每过多少代显示一次日志
def train(network, 
        train_loader, test_loader, 
        optimizer, inferencer, scheduler=None, warmup_scheduler=None,
        iteration_show=0, max_epoch=100):
    loss_fun = torch.nn.MSELoss()
    epoch_loss = []  # 每个epoch的平均loss
    epoch_val_loss_mean = []  # 每个epoch在测试集上的平均loss
    status = 0
    for i in range(max_epoch):
        network.train()  # 启用Batch Normalization 和 Dropout（仅当模型中有这俩玩意的时候有效！）
        status += 1
        avg_loss = 0  # 每个epoch中所有iteration的平均loss
        j = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            output = network(data)
            loss = loss_fun(output, target)
            cur_loss = loss.data.to('cpu')
            avg_loss += cur_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 每个iter的结果日志
            max_iteration = len(train_loader)
            if iteration_show == 0:
                iteration_show = int(max_iteration / 4)
            if (j + 1) % iteration_show == 0 or j + 1 == max_iteration:
                cur_lr = optimizer.state_dict()['param_groups'][0]['lr']
                print('Epoch [%s/%s][%s/%s]' % (i + 1, max_epoch, j + 1, max_iteration), 
                    ': loss:', cur_loss.item(), 
                    ', lr:', cur_lr)

            # 前(i * max_iteration + j)次iter warmup scheduler生效
            if warmup_scheduler != None and (i * max_iteration + j)  < warmup_scheduler.warmup_steps:
                warmup_scheduler.step()
            j += 1
        
        # 每个epoch按主scheduler更新lr
        if scheduler != None:
            scheduler.step()
        # 训练集损失
        avg_loss /= j
        epoch_loss.append(avg_loss)  # 按epoch打印loss
        print('avg loss =', avg_loss)  # 输出每个epoch中的平均loss

        # 验证集损失
        val_loss_mean = inferencer.validation(network, test_loader, i + 1)
        epoch_val_loss_mean.append(val_loss_mean)
        print('training completed:', int((status/max_epoch)*100), '%\n')
    
    conf_parser = cp.ConfigParser()
    draw.draw_2_data([epoch_loss, epoch_val_loss_mean], ['train', 'validation'], xlabel='epoch', ylabel='loss',
                        name="train&validation loss")
    saver.DataSaver().to_csv(epoch_loss, conf_parser.conf_dict['workdir']['result_dir'], 'train_loss_arr')
    saver.DataSaver().to_csv(epoch_val_loss_mean, conf_parser.conf_dict['workdir']['result_dir'], 'validation_loss_arr')


def test(network, testloader, inferencer):
    inferencer.test(network, testloader)


# 加载model的参数到一个【同质】的空网络变量（同质是指网络结构一模一样，或者说和磁盘中的模型是同一个类的实例）
def model_playback(network, testloader, inferencer, path):
    network = torch.load(path)  # 读入模型
    # network.load_state_dict(torch.load(path))  # 读入参数
    inferencer.test(network, testloader)

