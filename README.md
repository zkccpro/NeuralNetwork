# NeuralNetwork
一个易用的神经网络框架，宗旨是灵活搭建任意用途的神经网络。



## 1. 环境

python3.8

pytorch1.11/1.12

CUDA11.1+



## 2. 模块支持

### 2.1 Conv

### 2.2 FullConnection

### 2.3 Residual Block

### 2.4 Bottleneck Block

### 2.5 SE Block

### 2.6 Unet

## 3. 功能/特性支持

详细说明 及 接口规范 见`doc/`。

### 3.1 统一配置

工程运行时行为的配置文件，在`conf/`路径下。

项目使用.py配置文件风格：优点是易读性强、方便书写配置、解起来也容易；但缺点是 还没找到一种良好的配置文件读取方法。（用python内置的vars()函数读配置文件会有很多自定义变量，造成冗余...）

#### 3.1.1 conf/workdir.py

* **version：**since beta-0.5

输出路径相关配置。

#### 3.1.2 conf/globalParam.py

* **version：**since beta-0.1（计划在beta-0.7版本弃用）

包括 device配置、网络配置、优化器配置、学习率调度器配置、预处理接口。

#### 3.1.3 schedule

* **version：**计划在beta-0.7支持。

包括优化器配置、学习率调度器配置、使用设备配置。

#### 3.1.4 network

* **version：**计划在beta-0.7支持。

网络相关的各种设置。

#### 3.1.5 data

* **version：**计划在beta-0.7支持。

包括数据parser配置、预处理配置等。（预处理这块可能还需要一个类似parser的统一接口）



### 3.2 parsers

提供统一parser接口，在`parse/`路径下。用户可继承之实现自定义parser，用于解析配置文件、解析各种格式的输入数据等，解析后的数据将交给DataReader接口，其负责进一步封装交给DataSet：

`parsers->DataReader->DataSet`

暂不支持parser注册，需要在单测文件中调用`dataReader.xxxDataReader`函数时指定parser。

#### 3.2.1 config parser

* **version：**since beta-0.5

解工程配置文件到单例字典中。

#### 3.2.2 pictures parser

* **version：**since beta-0.5

解图片类型数据，返回python list给框架。

#### 3.2.2 Csv parser

* **version：**计划在beta-0.7支持。

解csv类型数据标签，返回python list给框架。

#### 3.2.2 Coco parser

* **version：**计划在beta-0.7支持。

解coco类型数据集标注，返回python list给框架。

#### 3.2.2 Voc parser

* **version：**计划在beta-0.7支持。

解voc类型数据集标注，返回python list给框架。



### 3.3 数据预处理接口

目前用户可在`conf/globalParam.py	`中自定义数据预处理函数，但还不支持配置化和注册。需要在单测文件中的`dataSet.CustomedDataSet`函数中指定。



### 3.4 推理接口

* **version：**since beta-0.5

提供通用的推理接口，可以继承推理接口以给不同任务定义不一样的推理(inference)、测试(test)、验证(validation)方法。

#### 3.4.1 ClassifyInference

* **version：**since beta-0.5

分类任务的推理策略。（还没完成）

#### 3.4.2 RegressionInference

* **version：**since beta-0.5

回归任务的推理策略。

#### 3.4.3 GennetInference

* **version：**since beta-0.5

生成任务的推理策略。



### 3.5 结果保存

* **version：**since beta-0.5

目前提供对 曲线图、生成图像、源数据 结果的保存。

#### 3.5.1 曲线图

* **version：**since beta-0.2

目前提供3种画图函数：`draw_1d`、`draw_2d`、`draw_2_data`。分别用来画一维曲线图、二维曲线图、2个数据的曲线图。可以在inference或action里调用。

暂时还没想好怎么把各个画图函数统一成一个接口...

#### 3.5.2 生成图像

* **version：**since beta-0.5

保存生成网络等生成的图像。可以在inference或action里调用。

#### 3.5.3 源数据

* **version：**since beta-0.5

提供一个保存tensor形式数据到csv文件的能力。可以在inference或action里调用。

#### 3.5.4 模型

* **version：**since beta-0.2

以保存完整模型的方式保存到.pth文件，beta-0.5之后提供统一接口。可以在inference或action里调用。

#### 3.5.4 日志

* **version：**计划 beta-0.7支持。



## 快速开始

1. 把你的图像数据放到`data/`路径下，并在DataReader中确保数据路径是正确的。

   

2. 检查单测文件中的执行计划及训练网络符合预期，把单测函数加入main.py：

   ```python
   from torch.utils.data import DataLoader
   from core import dataReader, dataSet, action
   from conf import globalParam
   from post_processing import inferencer
   
   
   def ut_TinyData_twostage():
       # 1. load data and label
       # 2w张图片，训练时稳定占用25G内存
       pic_list, test_list, label, test_label = dataReader.PredictDataReader([r"data/TinyData/"], None).read(default=True)
   
       # 2. prepare train data
       dataset = dataSet.CustomedDataSet(train_x=pic_list, train_y=label,
                                         transform=globalParam.pic_transfer, target_transform=globalParam.label_transfer,
                                         data_transform=globalParam.pic_data_transfer,
                                         target_data_transform=globalParam.num_data_transfer)
       dataloader = DataLoader(dataset, batch_size=21, shuffle=False, sampler=None,
                  batch_sampler=None, num_workers=0, collate_fn=None,
                  pin_memory=False, drop_last=False, timeout=0,
                  worker_init_fn=None)
   
       # 2. prepare test data
       testset = dataSet.CustomedDataSet(train=False, test_x=test_list, test_y=test_label,
                                         transform=globalParam.pic_transfer, target_transform=globalParam.label_transfer,
                                         data_transform=globalParam.pic_data_transfer,
                                         target_data_transform=globalParam.num_data_transfer)
       testloader = DataLoader(testset, batch_size=1, shuffle=False, sampler=None,
                  batch_sampler=None, num_workers=0, collate_fn=None,
                  pin_memory=False, drop_last=False, timeout=0,
                  worker_init_fn=None)
   
       # prepare model
       cnn = globalParam.twostage_network
       reg_inferencer = inferencer.RegressionInference()  # 推理器（后件处理接口）
   
       # train
       print('--------------------------------------------')
       print('start training!')
       action.train(cnn, dataloader, testloader,
                   globalParam.twostage_optimizer, reg_inferencer,
                   scheduler=globalParam.scheduler, warmup_scheduler=globalParam.warmup_scheduler,
                   iteration_show=100, max_epoch=100)
   
       # test
       print('--------------------------------------------')
       print('start testing!')
       action.test(cnn, testloader, reg_inferencer)
   
       # playback
       # print('--------------------------------------------')
       # print('playback model!')
       # action.model_playback(cnn, testloader, gennet_inferencer, conf_parser.conf_dict['workdir']['checkpoint_dir'] + 'epoch_100.pth')
   
   
   ```

3. 检查`globalParam.py`中的网络及训练参数符合预期：

   ```python
   # 网络
   ...
   # 优化器
   ...
   # 学习率调度器
   ...
   ```

4. 跑一下！
