# 一些便于调用的计算函数


# 用于求装有tensor的list的均值
def mean(lst):
    ret = 0
    for data in lst:
        ret += data
    return ret / len(lst)

