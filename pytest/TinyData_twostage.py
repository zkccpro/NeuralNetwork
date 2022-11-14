from torch.utils.data import DataLoader
from core import dataReader, dataSet, action
from conf import globalParam
from post_processing import inferencer
from parse import configParser as cp


# 数据集采用TinyData+DifPic，1w+1w张图片，每21张图片为一个batch，基于同一个图片调整的曝光值
def ut_TinyData_twostage():
    # 1. load data and label
    # 2w张图片，训练时稳定占用25G内存
    pic_list, test_list, label, test_label = dataReader.PredictDataReader([r"data/TinyData/"], None).read(50,42)

    # 2. prepare train data
    dataset = dataSet.CustomedDataSet(train_x=pic_list, train_y=label,
                                      transform=globalParam.pic_transfer, target_transform=globalParam.label_transfer,
                                      data_transform=globalParam.pic_data_transfer,
                                      target_data_transform=globalParam.num_data_transfer)
    dataloader = DataLoader(dataset, batch_size=21, shuffle=False, sampler=None,
               batch_sampler=None, num_workers=0, collate_fn=None,
               pin_memory=False, drop_last=False, timeout=0,
               worker_init_fn=None)

    # 2. prepare test data
    testset = dataSet.CustomedDataSet(train=False, test_x=test_list, test_y=test_label,
                                      transform=globalParam.pic_transfer, target_transform=globalParam.label_transfer,
                                      data_transform=globalParam.pic_data_transfer,
                                      target_data_transform=globalParam.num_data_transfer)
    testloader = DataLoader(testset, batch_size=1, shuffle=False, sampler=None,
               batch_sampler=None, num_workers=0, collate_fn=None,
               pin_memory=False, drop_last=False, timeout=0,
               worker_init_fn=None)

    # prepare model
    cnn = globalParam.twostage_network
    reg_inferencer = inferencer.RegressionInference()  # 推理器（后件处理接口）
    conf_parser = cp.ConfigParser()

    # train
    print('--------------------------------------------')
    print('start training!')
    action.train(cnn, dataloader, testloader,
                globalParam.twostage_optimizer, reg_inferencer,
                scheduler=globalParam.scheduler, warmup_scheduler=globalParam.warmup_scheduler,
                iteration_show=100, max_epoch=2)

    # test
    print('--------------------------------------------')
    print('start testing!')
    action.test(cnn, testloader, reg_inferencer)

    # playback
    print('--------------------------------------------')
    print('playback model!')
    action.model_playback(cnn, testloader, reg_inferencer, conf_parser.conf_dict['workdir']['checkpoint_dir'] + 'epoch_2.pth')
