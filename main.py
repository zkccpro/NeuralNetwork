import os
import utility
import test.TinyData_cnn
import test.TinyData_cnn_doubleinput
import test.TinyData_unet


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 见https://blog.csdn.net/m0_50736744/article/details/121799432


if __name__ == '__main__':
    if os.path.exists("output/"):
        utility.rm_rf("output/")
    os.mkdir("output/")
    # test.RandomData_nn.ut_nn()
    # test.RandomData_nn.ut_nn_without_dataloader()
    test.TinyData_cnn.ut_TinyData_cnn()
    # test.TinyData_cnn_doubleinput.ut_TinyData_cnn_doubleinput()
    # test.TinyData_unet.ut_TinyData_unet()
    print('hello my fist NN model!')


