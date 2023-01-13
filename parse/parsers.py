import os
from PIL import Image
from core import utility as util


class ParserInterface:
    def __init__(self):
        pass

    def parse(path):
        pass


@util.Singleton
class PicturesParser(ParserInterface):
    def __init__(self):
        pass
    
    def parse(self, path, total_num=0, train_num=0, 
            contain_tag=False, min_stat=20):
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
        for pic in os.listdir(path):
            if i >= total_num:
                break
            if min_stat != 0 and i % stat_every == 1:  # 显示进度
                stat = int((i / total_num) * 100)
                print('loading data completed:', stat, '%')
            img = Image.open(os.path.join(path, pic)).convert('L')
            if contain_tag:
                tag = float(pic[len(pic) - 8: len(pic) - 4])

            if i <= train_num:  # 训练
                pic_list.append(img)
                if contain_tag:
                    label.append(tag)
            else:  # 测试
                test_list.append(img)
                if contain_tag:
                    test_label.append(tag)
            i += 1
        return pic_list, test_list, label, test_label


@util.Singleton
class CsvParser(ParserInterface):
    def __init__(self):
        pass
    
    def parse():
        pass


@util.Singleton
class JsonParser(ParserInterface):
    def __init__(self):
        pass
    
    def parse():
        pass


@util.Singleton
class JsonPathParser(ParserInterface):
    def __init__(self):
        pass
    
    def parse(label_path, total_num, train_num):
        pass


@util.Singleton
class XmlParser(ParserInterface):
    def __init__(self):
        pass
    
    def parse():
        pass


@util.Singleton
class XmlPathParser(ParserInterface):
    def __init__(self):
        pass
    
    def parse(label_path, total_num, train_num):
        pass