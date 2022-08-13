import os
from PIL import Image


# 对外（单测文件）暴露的接口，包括一个读数据的方法和一个数据路径
class DataReaderInterface:
    def __init__(self, path):
        self.path = path
        pic_num = 0
        for _, _, filenames in os.walk(path):
            pic_num = len(filenames)
        self.path_pic_num = pic_num

    # 读数据方法
    # total_num：训练+测试的所有数据
    # train_num：参与训练的所有数据
    # default=True时：采用路径中所有图片进行训练+测试，测试集规模为500
    # min_stat：最小显示进度的分度，为0时不显示进度，默认为20，即 每读取20分之一的图片显示一次进度
    # 返回值：{pic_list, test_list, label, test_label}四元组
    def read(self, total_num=0, train_num=0, default=False, min_stat=20):
        pass


# ------------整合层，整合读盘层读到的数据--------------
# data：三维（二维图像数组），额外填充了一维，实际上是四维
# label：一维（数组）
class DataReader(DataReaderInterface):
    def __init__(self, path):
        super(DataReader, self).__init__(path)

    def read(self, total_num=0, train_num=0, default=False, min_stat=20):
        # 读图像和标签
        pic_list, test_list, label, test_label = PictureReader(self.path).read(total_num, train_num, default, min_stat)
        # 升维
        for i in range(len(pic_list)):
            pic_list[i] = [pic_list[i]]
        for i in range(len(test_list)):
            test_list[i] = [test_list[i]]
        return pic_list, test_list, label, test_label


# 分别读取组两组图像组成双通道的输入，用于两个输入图片的网络（两路输入网络）
class DataReaderDouble(DataReaderInterface):
    def __init__(self, path, path_2):
        super(DataReaderDouble, self).__init__(path)
        self.path_2 = path_2

    def read(self, total_num=0, train_num=0, default=False, min_stat=20):
        pic_list = []
        test_list = []
        # 读图像和标签
        src_list, src_test_list, _, _ = PictureReader(self.path).read(total_num, train_num, default, min_stat)
        diff_list, diff_test_list, label, test_label = PictureReader(self.path_2).read(total_num, train_num, default,
                                                                                       min_stat)
        # 升维
        for i in range(len(src_list)):
            pic_list.append([src_list[i], diff_list[i]])
        for i in range(len(src_test_list)):
            test_list.append([src_test_list[i], diff_test_list[i]])

        return pic_list, test_list, label, test_label


# 读取两组图片分别作为data和label，用于读取Unet等类型的网络数据
class UnetDataReader(DataReaderInterface):
    def __init__(self, path, path_2):
        super(UnetDataReader, self).__init__(path)
        self.path_2 = path_2

    def read(self, total_num=0, train_num=0, default=False, min_stat=20):
        pic_list, test_list, _, _ = PictureReader(self.path).read(total_num, train_num, default, min_stat)
        label, test_label, _, _ = PictureReader(self.path_2).read(total_num, train_num, default, min_stat)
        # 升维
        for i in range(len(pic_list)):
            pic_list[i] = [pic_list[i]]
        for i in range(len(test_list)):
            test_list[i] = [test_list[i]]
        return pic_list, test_list, label, test_label


# --------------读盘层，直接读取各种形式的磁盘数据------------------
# 读图片的具象
# 这里直接读取了图像文件名中隐含的标签信息，当标签是单个数字时建议把标签放在图像文件名中
class PictureReader(DataReaderInterface):
    def __init__(self, path):
        super(PictureReader, self).__init__(path)

    def read(self, total_num=0, train_num=0, default=False, min_stat=20):
        if default:
            total_num = self.path_pic_num
            train_num = total_num - 500
        if total_num <= train_num:
            raise ValueError("total_num <= train_num!")
        pic_list = []
        test_list = []
        label = []
        test_label = []
        # 读图片、标签
        i = 0
        stat_every = 0
        if min_stat != 0:
            stat_every = int(total_num / min_stat)
        print("starting reading data, total num =", total_num, ", train_num =", train_num)
        for pic in os.listdir(self.path):
            if i >= total_num:
                break
            if min_stat != 0 and i % stat_every == 1:  # 显示进度
                stat = int((i / total_num) * 100)
                print('loading data completed:', stat, '%')
            img = Image.open(os.path.join(self.path, pic)).convert('L')
            tag = float(pic[len(pic) - 8: len(pic) - 4])
            # 提升维度，否则训练时会把batch当做第三维度
            if i <= train_num:  # 训练
                pic_list.append(img)
                label.append(tag)
            else:  # 测试
                test_list.append(img)
                test_label.append(tag)
            i += 1
        return pic_list, test_list, label, test_label


# 读txt文件的具象
class TxtReader(DataReaderInterface):
    def __init__(self, path):
        super(TxtReader, self).__init__(path)

    def read(self, total_num=0, train_num=0, default=False, min_stat=20):
        pass


# 读csv文件的具象
class CsvReader(DataReaderInterface):
    def __init__(self, path):
        super(CsvReader, self).__init__(path)

    def read(self, total_num=0, train_num=0, default=False, min_stat=20):
        pass


# 随机产生一组数据的具象
class RandomReader(DataReaderInterface):
    def __init__(self, path):
        super(RandomReader, self).__init__(path)

    def read(self, total_num=0, train_num=0, default=False, min_stat=20):
        pass

