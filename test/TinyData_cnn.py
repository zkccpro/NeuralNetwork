import os
from PIL import Image
from torch.utils.data import DataLoader
import handleData
import globalParam
import action
import model


# 数据集采用TinyData，1w张图片，每21张图片为一个batch，基于同一个图片调整的曝光值
def ut_TinyData_cnn():
    # 1. load data and label
    # 读入数据
    pic_list, test_list, label, test_label = handleData.DataReader(r"data/TinyData/").read(42, 10)
    # path = r"data/TinyData/"
    # pic_num = 0  # 参与训练样本数量
    # for _, _, filenames in os.walk(path):
    #     pic_num = len(filenames)
    # print('loading data from disk! total data:', pic_num)
    #
    # pic_list = []
    # test_list = []
    # label = []
    # test_label = []
    # i = 0
    # # 读图片、标签
    # for pic in os.listdir(path):
    #     if i % 500 == 50:  # 显示进度
    #         stat = int((i / pic_num) * 100)
    #         print('loading data completed:', stat, '%')  # python这个print是有问题的，不要想着用它打日志了
    #         break  # 这里测试单个样本的时候可以break掉，以免加载过多样本
    #     img = Image.open(os.path.join(path, pic)).convert('L')
    #     img.info["name"] = pic  # Image对象里有个info字典，可以往这里加一些自己的信息
    #     tag = float(pic[len(pic) - 8: len(pic) - 4])
    #     # print('picture name is:', pic, 'tag is:', tag)
    #     # 这里细节需要十分注意！最外层一定要加一个[]，否则后面训练会崩，为啥呢？
    #     # 你前面是把一个单通道图像转换成np array，就导致train_x[i]只有二维！这样就会导致batch>1的时候，
    #     # dataloader会把多个二维的图像叠成一个batch【通道】的图像，就造成本来应该是一个四维tensor变成一个三维tensor
    #     # 你设置的batch数变成了通道数，正确做法就是在这里给train_x[i]添加一维，变成三维的单通道np array
    #     if i <= 42:  # 训练
    #         pic_list.append([img])
    #         label.append(tag)
    #     else:  # 测试
    #         test_list.append([img])
    #         test_label.append(tag)
    #     i += 1

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
    cnn = globalParam.bottleneckcnn_network

    # train
    print('--------------------------------------------')
    print('start training!')
    action.train(cnn, dataloader, testloader, globalParam.bottleneckcnn_optimizer, epoch=21)

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

