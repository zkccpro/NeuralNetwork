from matplotlib import pyplot as plt
from parse import configParser as cp
import torch


# 输入：arr类型为list[tensor]
# 作用：画一维曲线图，一般用这个函数画逐代损失曲线、各个图片（0-10000）的网络输出（评分）等
# 注意：tensor必须得是一维的，二维及以上的tensor plot处理不了！
# 横轴根据输入arr的长度产生从零开始的序列
# 如果loss暴nan，横坐标的数字会不是整数
def draw_1d(arr, xlabel="x", ylabel="y", name="plt1", output_dir='result_dir'):
    conf_parser = cp.ConfigParser()
    x = []
    for i in range(arr.__len__()):
        x.append(i + 1)
    plt.plot(x, arr)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(conf_parser.conf_dict['workdir'][output_dir] + name + ".jpg", dpi=1200)
    plt.delaxes()
    print('plots saved...')


# 输入：arr_x，横轴数据，类型为list[tensor]；arr_y，纵轴数据，类型为list[tensor]
# 作用：画二维曲线图，一般用这个函数画输入网络数据和网络输出数据的关系
# 注意：tensor必须得是一维的，二维及以上的tensor plot处理不了！
def draw_2d(arr_x, arr_y):
    plt.plot(arr_x, arr_y)
    plt.delaxes()
    print('plots saved...')


# 把多个折线图画在一起，一般用于把ground_truth和predict_out折线画在一起
# arrs: 多个待绘制数据（1维tensor）的列表
# label: arrs中每个数据（arr）对应的名称label（数量需与arrs中的arr数量保持一致）
# todo: 这个函数会弹警告，以后需要处理一下
def draw_2_data(arrs, label, xlabel="x", ylabel="y", name="plt2", output_dir='result_dir'):
    conf_parser = cp.ConfigParser()
    x = []
    line_graph = []
    for i in range(len(arrs[0])):
        i = i + 1
        x.append(i)
    for i in range(len(arrs)):
        tmp, = plt.plot(x, arrs[i], label=label[i])
        line_graph.append(tmp)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(handles=line_graph, labels=label)
    plt.savefig(conf_parser.conf_dict['workdir'][output_dir] + name + ".jpg", dpi=1200)
    plt.delaxes()
    print('plots saved...')
