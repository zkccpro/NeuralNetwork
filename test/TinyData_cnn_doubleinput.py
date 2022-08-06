import os
from PIL import Image
from torch.utils.data import DataLoader
import handleData
import globalParam
import action
import model
import torch


# 数据集采用TinyData，1w张图片，每21张图片为一个batch，基于同一个图片调整的曝光值
def ut_TinyData_cnn_doubleinput():
    # x = torch.randn(5, 2, 3, 4)
    # print(x)
    # x1, x2 = x.split(1, 1)
    # print(x1.shape, x2.shape)  # x1,x2:[5, 1, 3, 4]
    # print(x1, x2)
    #
    # xn = torch.cat([x1, x2], dim=1)  # 在1维度（从左往右数），合并x1、x2
    # print(xn.shape)  # xn:[5, 2, 3, 4]
    # print(xn)

    # 1. load data and label
    # 读入数据
    pic_list, test_list, label, test_label = handleData.DataReaderDouble(r"data/TinyData/", r"data/DiffPic/").read(default=True)

    # 2. prepare train data
    dataset = handleData.CustomedDataSet(train_x=pic_list, train_y=label, transform=globalParam.pic_transfer, target_transform=globalParam.label_transfer)
    dataloader = DataLoader(dataset, batch_size=21, shuffle=False, sampler=None,
               batch_sampler=None, num_workers=0, collate_fn=None,
               pin_memory=False, drop_last=False, timeout=0,
               worker_init_fn=None)

    # 2. prepare test data
    testset = handleData.CustomedDataSet(train=False, test_x=test_list, test_y=test_label, transform=globalParam.pic_transfer, target_transform=globalParam.label_transfer)
    testloader = DataLoader(testset, batch_size=1, shuffle=False, sampler=None,
               batch_sampler=None, num_workers=0, collate_fn=None,
               pin_memory=False, drop_last=False, timeout=0,
               worker_init_fn=None)

    # prepare model
    cnn = globalParam.bottleneckcnn_doubleinput_network

    # train
    print('--------------------------------------------')
    print('start training!')
    action.train(cnn, dataloader, testloader, globalParam.bottleneckcnn_doubleinput_optimizer, epoch=21)

    # test
    print('--------------------------------------------')
    print('start testing!')
    action.test_regression(cnn, testloader, draw_or_not=True)

    # # save，不要轻易save，容易冲掉以前的模型！
    # print('--------------------------------------------')
    # print('saving model!')
    # action.save_model(cnn, 'model/ResidualCNN/neck_batch=21_epoch=100_lr=0.00002_momentum=0.9_weight_decay=1e-2')
    #
    # # playback
    # print('--------------------------------------------')
    # print('playback model!')
    # playback_cnn = model.BottleneckCNN()  # 这里记得改！
    # action.model_playback(playback_cnn, 'model/ResidualCNN/neck_batch=21_epoch=100_lr=0.00002_momentum=0.9_weight_decay=1e-2')
    # action.test_regression(playback_cnn, testloader, draw_or_not=True)

