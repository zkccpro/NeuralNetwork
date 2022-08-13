from torch.utils.data import DataLoader
import dataReader
import dataSet
import globalParam
import action


# 数据集采用TinyData+DifPic，1w+1w张图片，每21张图片为一个batch，基于同一个图片调整的曝光值
def ut_TinyData_cnn_doubleinput():
    # 1. load data and label
    # 2w张图片，训练时稳定占用25G内存
    pic_list, test_list, label, test_label = dataReader.DataReaderDouble(r"data/TinyData/", r"data/DiffPic/").read(50, 42)

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
    cnn = globalParam.secnn_doubleinput_network

    # train
    print('--------------------------------------------')
    print('start training!')
    action.train(cnn, dataloader, testloader, globalParam.secnn_doubleinput_optimizer, epoch=5)

    # test
    print('--------------------------------------------')
    print('start testing!')
    action.test_regression(cnn, testloader, draw_or_not=True)

    # # save，不要轻易save，容易冲掉以前的模型！
    # print('--------------------------------------------')
    # print('saving model!')
    # action.save_model(cnn, 'model/ResidualCNN/doubleinput_se_batch=21_epoch=100_lr=0.00005_momentum=0.9_weight_decay=1e-2')
    #
    # # playback
    # print('--------------------------------------------')
    # print('playback model!')
    # playback_cnn = model.SECNNDoubleInput()  # 这里记得改！
    # action.model_playback(playback_cnn, 'model/ResidualCNN/doubleinput_se_batch=21_epoch=100_lr=0.00005_momentum=0.9_weight_decay=1e-2')
    # action.test_regression(playback_cnn, testloader, draw_or_not=True)

