import os
import sys
o_path = os.getcwd()  # 把工程根路径添加到python环境变量
sys.path.append(o_path)
import utility
from pytest import TinyData_unet, TinyData_cnn_doubleinput, TinyData_cnn, TinyData_twostage
from conf import workdir as wd
from parse import configParser as cp


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 见https://blog.csdn.net/m0_50736744/article/details/121799432

if __name__ == '__main__':
    conf_parser = cp.ConfigParser()
    conf_parser.parse()
    # print(
    #     'YOUR CONFIGS ARE:\n----------------------------------------------\n',
    #     cp.ConfigParser().parse(),
    #     '\n----------------------------------------------\n'
    # )
    if not os.path.exists('output/'):
        os.mkdir('output/')
    if not os.path.exists(conf_parser.conf_dict['workdir']['work_dir']):
        os.mkdir(conf_parser.conf_dict['workdir']['work_dir'])
    if os.path.exists(conf_parser.conf_dict['workdir']['ts']):
        utility.rm_rf(conf_parser.conf_dict['workdir']['ts'])
    os.mkdir(conf_parser.conf_dict['workdir']['ts'])
    os.mkdir(conf_parser.conf_dict['workdir']['checkpoint_dir'])
    os.mkdir(conf_parser.conf_dict['workdir']['log_dir'])
    os.mkdir(conf_parser.conf_dict['workdir']['result_dir'])

    TinyData_cnn.ut_TinyData_cnn()
    TinyData_cnn_doubleinput.ut_TinyData_cnn_doubleinput()
    TinyData_unet.ut_TinyData_unet()
    TinyData_twostage.ut_TinyData_twostage()
    print('hello my fist NN model!')

