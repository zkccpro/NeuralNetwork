import torch
from torchvision import transforms
import csv
import PIL 

# 保存接口
class SaverInterface:
    def __init__(self):
        pass

    # 直接把文件存到磁盘的指定路径，比如图像文件、模型文件等
    def to_disk(self, file, path, name):
        pass

    # 把数据数组存到指定路径的csv文件
    # 数据数组是一个[tensor]（tensor可能多维）
    def to_csv(self, arr, path, name):
        pass


# 保存[tensor]中的数据
class DataSaver(SaverInterface):
    def __init__(self):
        super(DataSaver).__init__()

    def to_csv(self, arr, path, name):
        for i in range(len(arr)):
            # 先把list中的每个tensor转成list
            if arr[i].dim() > 0:  # 这里考虑了tensor维度大于1的情况，比如多损失模型
                arr[i] = arr[i].tolist()
            else:
                arr[i] = [arr[i].tolist()]
        with open(path + name + '.csv', 'w') as file:
            writer = csv.writer(file)
            writer.writerows(arr)
        print('arr data saved...')


# 保存一张图片（后缀.jpg）
class ImgSaver(SaverInterface):
    def __init__(self):
        super(ImgSaver).__init__()

    # file是一个tensor
    def to_disk(self, file, path, name):
        pic = transforms.ToPILImage()(file.squeeze(0))
        pic.save(path + name + '.jpg')


# 保存一个模型（后缀.pth）
class ModelSaver(SaverInterface):
    def __init__(self):
        super(ModelSaver).__init__()

    def to_disk(self, file, path, name):
        torch.save(file, path + name + '.pth')  # 保存整个模型
        # torch.save(file.state_dict(), path + name + '.pth')  # 保存参数
        print('checkpoint saved')

