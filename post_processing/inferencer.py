import torch
from post_processing import draw
from post_processing import saver
from parse import configParser as cp
from core import utility
import time


# 各种不同的测试验证方法的action暴露的接口
class TestValidationInterface:
    def __init__(self):
        self.conf_parser = cp.ConfigParser()

    # 训练后的测试方法
    # network: 测试用的模型
    # testloader：测试样本
    def test(self, network, testloader):
        pass

    # 训练过程中的验证方法
    def validation(self, network, testloader, epoch):
        print('start validating in epoch', epoch, ':')



# 分类网络用的验证和测试
class ClassifyInference(TestValidationInterface):
    def __init__(self):
        super().__init__()

    def inference(self, network, testloader):
        pass

    def test(self, network, testloader):
        super().test(network, testloader)
        for data, target in testloader:
            outputs = network(data)
            _, predicted = torch.max(outputs.data, 1)  # torch.max(data,1)：返回tensor中每行的最大值和索引
        print("prediction is: ", predicted)  # 这个输出要按你自己想要的调整的，这个输出是预测最大可能的数字

    def validation(self, network, testloader, epoch):
        super().validation(network, testloader, epoch)


# 回归网络用的验证和测试
class RegressionInference(TestValidationInterface):
    def __init__(self):
        super().__init__()

    def inference(self, network, testloader):
        network.eval()
        predict_arr = []
        target_arr = []
        loss_arr = []
        loss_fun = torch.nn.MSELoss()
        T1 = time.perf_counter()
        image_num = 0
        for data, target in testloader:
            outputs = network(data)
            loss = loss_fun(outputs, target)
            loss_arr.append(loss.data.to('cpu'))
            predict_arr.append(outputs.data.to('cpu'))
            target_arr.append(target.data.to('cpu'))
            image_num += 1
        T2 = time.perf_counter()
        print('inference', image_num , 'images in: %sms' % ((T2 - T1)*1000))
        return predict_arr, target_arr, loss_arr

    def test(self, network, testloader):
        super().test(network, testloader)
        predict_arr, target_arr, loss_arr = self.inference(network, testloader)
        # draw
        draw.draw_1d(loss_arr, xlabel='label', ylabel='loss', name="test_loss")
        draw.draw_2_data([target_arr, predict_arr], ['target', 'predict'], xlabel='label', ylabel='value',
                        name="test_ret")
        # save
        saver.DataSaver().to_csv(loss_arr, self.conf_parser.conf_dict['workdir']['result_dir'], 'loss_arr')
        saver.DataSaver().to_csv(predict_arr, self.conf_parser.conf_dict['workdir']['result_dir'], 'predict_arr')
        saver.DataSaver().to_csv(target_arr, self.conf_parser.conf_dict['workdir']['result_dir'], 'target_arr')

    def validation(self, network, testloader, epoch):
        super().validation(network, testloader, epoch)
        predict_arr, target_arr, loss_arr = self.inference(network, testloader)
        # log
        val_loss_mean = utility.mean(loss_arr)  # test_loader为空这里弹warning，test_loss暴nan这里弹warning
        print('test loss mean =', val_loss_mean)  # 输出该epoch训练后在测试集上的平均损失

        # save model
        saver.ModelSaver().to_disk(network, self.conf_parser.conf_dict['workdir']['checkpoint_dir'], 'epoch_' + str(epoch))
        return val_loss_mean


# 生成网络用的验证和测试
class GennetInference(TestValidationInterface):
    def __init__(self):
        super().__init__()

    def inference(self, network, testloader):
        network.eval()
        predict_arr = []
        target_arr = []
        loss_arr = []
        loss_fun = torch.nn.MSELoss()
        T1 = time.perf_counter()
        image_num = 0
        for data, target in testloader:
            outputs = network(data)
            loss = loss_fun(outputs, target)
            loss_arr.append(loss.data.to('cpu'))
            predict_arr.append(outputs.data.to('cpu'))
            target_arr.append(target.data.to('cpu'))
            image_num += 1
        T2 = time.perf_counter()
        print('inference', image_num , 'images in: %sms' % ((T2 - T1)*1000))
        return predict_arr, target_arr, loss_arr

    def test(self, network, testloader):
        super().test(network, testloader)
        predict_arr, target_arr, loss_arr = self.inference(network, testloader)
        # draw
        draw.draw_1d(loss_arr, xlabel='label', ylabel='loss', name="test_loss")
        # save
        saver.DataSaver().to_csv(loss_arr, self.conf_parser.conf_dict['workdir']['result_dir'], 'test_loss_arr')
        for i in range(len(predict_arr)):
            # 这里把tensor先压缩成1维然后再转PIL（应该是已经考虑归一化的问题了？）
            saver.ImgSaver().to_disk(predict_arr[i], self.conf_parser.conf_dict['workdir']['result_dir'], 'predict_' + str(i))
            saver.ImgSaver().to_disk(target_arr[i], self.conf_parser.conf_dict['workdir']['result_dir'], 'target_' + str(i))


    def validation(self, network, testloader, epoch):
        super().validation(network, testloader, epoch)
        predict_arr, target_arr, loss_arr = self.inference(network, testloader)
        # log
        val_loss_mean = utility.mean(loss_arr)  # test_loader为空这里弹warning，test_loss暴nan这里弹warning
        print('test loss mean =', val_loss_mean)  # 输出该epoch训练后在测试集上的平均损失

        # save model
        saver.ModelSaver().to_disk(network, self.conf_parser.conf_dict['workdir']['checkpoint_dir'], 'epoch_' + str(epoch))
        return val_loss_mean

