# 一些便于调用的计算函数
import os

# 用于求装有tensor的list的均值
def mean(lst):
    ret = 0
    for data in lst:
        ret += data
    return ret / len(lst)


# 递归删除路径，相当于rm -rf xxx/
def rm_rf(path_data):
    for i in os.listdir(path_data):  # os.listdir(path_data)  # 返回一个列表，里面是当前目录下面的所有东西的相对路径
        file_data = path_data + i  # 当前文件夹的下面的所有东西的绝对路径
        if os.path.isfile(file_data):  # os.path.isfile判断是否为文件,如果是文件,就删除.如果是文件夹.递归给del_file.
            os.remove(file_data)
        else:
            rm_rf(file_data)
    os.rmdir(path_data)

