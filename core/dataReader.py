import os
from PIL import Image
from parse import parsers


# 对外（单测文件）暴露的接口，包括一个读数据的方法和一个数据路径
# 这个接口将各种格式的文件（data/label）读到最终dataset可直接使用的数组
class DataReaderInterface:
    # 
    def __init__(self, data_path_l, label_path):
        self.data_path_l = data_path_l  # 多个图片路径列表
        self.label_file = None  # 单个label文件
        self.label_path = None  # 含有多个label文件的路径
        self.fetch_num = 0  # fetch到的数据个数

        # fetch data: 检查data_path_l中各个路径下的文件数，如果不一致则ERROR
        data_num = 0
        for data_path in data_path_l:
            cur_num = 0
            for _, _, filenames in os.walk(data_path):
                cur_num = len(filenames)
            if cur_num != data_num and data_num > 0:
                print('ERROR: channels of data does not have same number!')
                raise ValueError("channels of data does not have same number!")
            data_num = cur_num
            print('fetched data in', data_path, 'and found data num:', cur_num)
        self.fetch_num = data_num

        # fetch label：检查输入的label_path是路径还是单个文件
        if label_path == None:
            print('WARNING: cannot find label path, you must have set label in picture name')
        elif os.path.isdir(label_path):
            # 是路径的话检查label文件个数与data个数是否相等
            self.label_path = label_path
            for _, _, filenames in os.walk(data_path):
                if self.fetch_num != len(filenames):
                    print('ERROR: data number does not eval to label number')
                    raise ValueError("data number does not eval to label number!")
        else:
            self.label_file = label_path
        print('fetched label in', label_path)

    # 读数据方法
    # total_num：训练+测试的所有数据
    # train_num：参与训练的所有数据
    # default=True时：采用路径中所有图片进行训练+测试，测试集规模为500
    # min_stat：最小显示进度的分度，为0时不显示进度，默认为20，即 每读取20分之一的图片显示一次进度
    # 返回值：{pic_list, test_list, label, test_label}四元组
    def read(self, total_num=0, train_num=0, default=False, min_stat=20):
        pass


# 分别读取组两组图像组成双通道的输入，用于两个输入图片的网络（两路输入网络）
class PredictDataReader(DataReaderInterface):
    def __init__(self, data_path_l, label_path, label_parser=None):
        super().__init__(data_path_l, label_path)
        self.label_parser = label_parser

    def read(self, total_num=0, train_num=0, default=False, min_stat=20):
        train_list = []
        test_list = []
        train_label = []
        test_label = []
        # 下面这俩要分别填到train_list和test_list肚子里的
        train_item = []
        test_item = []
        pic_parser = parsers.PicturesParser()

        # 处理默认训练规模
        if default:
            total_num = self.fetch_num
            train_num = total_num - 504  # todo: 这里先写死了，为了保证训练集能被batch_size整除
        if total_num <= train_num:
            print('ERROR: total_num must > train_num')
            raise ValueError("total_num must > train_num!")

        # parse图像和标签
        pic_tag = False
        # case1: 入参没有标签，认为标签在图片名字里
        if self.label_path == None and self.label_file == None:
            pic_tag = True
            i = 0
            for data_path in self.data_path_l:
                if i == 0:
                    train, test, train_label, test_label = pic_parser.parse(data_path, total_num, train_num, contain_tag=pic_tag, min_stat=min_stat)
                else:
                    train, test, _, _ = pic_parser.parse(data_path, total_num, train_num, contain_tag=pic_tag, min_stat=min_stat)
                train_item.append(train)
                test_item.append(test)
                i += 1
        # case2: 入参含有标签，且未指定label_parser
        elif self.label_parser == None:
            print('ERROR: label does not in picture name while no given label parser')
            raise ValueError("label does not in picture name while no given label parser!")
        # case4: 标签在一个路径下的若干文件里（一个标签文件对应一个数据）
        elif self.label_path != None:
            for data_path in self.data_path_l:
                train, test, _, _ = pic_parser.parse(data_path, total_num, train_num, contain_tag=pic_tag, min_stat=min_stat)
                train_item.append(train)
                test_item.append(test)
            train_label, test_label = self.label_parser(self.label_path, total_num, train_num)
        # case5: 所有标签在一个文件里
        else:  # self.label_file != None
            for data_path in self.data_path_l:
                train, test, _, _ = pic_parser.parse(data_path, total_num, train_num, contain_tag=pic_tag, min_stat=min_stat)
                train_item.append(train)
                test_item.append(test)
            train_label, test_label = self.label_parser(self.label_file, total_num, train_num)

        
        # 检查训练集和测试集的通道数是否为空
        if len(train_item) == 0 or len(test_item) == 0:
            print('ERROR: trainset or testset channel is 0')
            raise ValueError("trainset or testset channel is 0!")
        # 升维，确保channel占据第3维
        for i in range(len(train_item[0])):
            tmp_l = []
            for item in train_item:
                tmp_l.append(item[i])
            train_list.append(tmp_l)
        for i in range(len(test_item[0])):
            tmp_l = []
            for item in test_item:
                tmp_l.append(item[i])
            test_list.append(tmp_l)

        return train_list, test_list, train_label, test_label


# 读取两组图片分别作为data和label，用于读取生成网络的数据和标签
class GennetDataReader(DataReaderInterface):
    def __init__(self, data_path_l, label_path):
        super().__init__(data_path_l, label_path)

    def read(self, total_num=0, train_num=0, default=False, min_stat=20):
        train_list = []
        test_list = []
        train_label = []
        test_label = []
        # 下面这俩要分别填到train_list和test_list肚子里的
        train_item = []
        test_item = []
        pic_parser = parsers.PicturesParser()

        # 处理默认训练规模
        if total_num <= train_num:
            print('ERROR: total_num must > train_num')
            raise ValueError("total_num must > train_num!")
        if default:
            total_num = self.fetch_num
            train_num = total_num - 504  # todo: 这里先写死了，为了保证训练集能被batch_size整除

        for data_path in self.data_path_l:
            train, test, _, _ = pic_parser.parse(data_path, total_num, train_num, min_stat=min_stat)
            train_item.append(train)
            test_item.append(test)
        train_label, test_label, _, _ = pic_parser.parse(data_path, total_num, train_num, min_stat=min_stat)

        # 检查训练集和测试集的通道数是否为空
        if len(train_item) == 0 or len(test_item) == 0:
            print('ERROR: trainset or testset channel is 0')
            raise ValueError("trainset or testset channel is 0!")
        # 升维，确保channel占据第3维
        for i in range(len(train_item[0])):
            tmp_l = []
            for item in train_item:
                tmp_l.append(item[i])
            train_list.append(tmp_l)
        for i in range(len(test_item[0])):
            tmp_l = []
            for item in test_item:
                tmp_l.append(item[i])
            test_list.append(tmp_l)

        return train_list, test_list, train_label, test_label

