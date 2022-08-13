# NeuralNetwork
一个易用的神经网络框架，宗旨是便于搭建任意的神经网络。



## 环境

python3.8

pytorch1.11/1.12

CUDA11.6/11.7



## 支持

* NN

* CNN

* Residual Block：

  ```python
  # module.py
  # 残差模块
  class ResidualBlock(nn.Module):
      def __init__(self, input_channels, output_channels):
          super(ResidualBlock, self).__init__()
          self.input_channels = input_channels
          self.output_channels = output_channels
  
          self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1)
          self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1)
          # 旁路卷积，卷积核为1，控制通道改变，不对图像自身产生任何变化
          self.conv_side = nn.Conv2d(input_channels, output_channels, kernel_size=1)
  
      def forward(self, x):
          y = F.relu(self.conv1(x))
          y = self.conv2(y)
        if self.input_channels != self.output_channels:
            x = self.conv_side(x)
          return F.relu(x + y)
  ```

* Bottleneck Block：

  ```python
  # module.py
  # 瓶颈模块
  class BottleneckBlock(nn.Module):
      def __init__(self, input_channels, output_channels, low_channels):
          super(BottleneckBlock, self).__init__()
          self.input_channels = input_channels
          self.output_channels = output_channels
          self.low_channels = low_channels
  
          self.conv1 = nn.Conv2d(input_channels, low_channels, kernel_size=1)
          self.conv2 = nn.Conv2d(low_channels, low_channels, kernel_size=3, padding=1)
          self.conv3 = nn.Conv2d(low_channels, output_channels, kernel_size=1)
          # 旁路卷积，卷积核为1，控制通道改变，不对图像自身产生任何变化
          self.conv_side = nn.Conv2d(input_channels, output_channels, kernel_size=1)
  
      def forward(self, x):
          y = F.relu(self.conv1(x))
          y = F.relu(self.conv2(y))
          y = self.conv3(y)
        if self.input_channels != self.output_channels:
            x = self.conv_side(x)
          return F.relu(x + y)
  ```

  

## 路径说明

`data/`：存放待训练测试的图像数据。

`model/`：存放训练好的model。

`test/`：存放单测文件。



## 接口说明

1. `globalParam.py`：

   * 训练设备——cpu or cuda？
   * 网络——在这里实例化你搭建的网络模型。
   * 优化器——配置每个网络的优化器及训练参数等。
   * 图像预处理函数——用于可能的图像增强、预处理等任务，在加载数据时执行。

2. `handleData.py-DataReaderInterface`：

   ```python
   # 从磁盘读数据的接口，包括一个读数据的方法和一个数据路径。
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
   ```

3. `handleData.py-CustomedDataSet`：

   集成调整数据格式，供DataLoader做训练前的最终处理。

   

## 快速开始

1. 把你的图像数据放到`data/`路径下，并在DataReader中确保数据路径是正确的：

   ```python
   # test/TinyData_cnn_doubleinput.py  
   pic_list, test_list, label, test_label = handleData.DataReaderDouble(r"data/TinyData/", r"data/DiffPic/").read(default=True)
   ```

2. 检查单测文件中的执行计划及训练网络符合预期：

   ```python
   def ut_TinyData_cnn_doubleinput():
       # 1. load data and label
       pic_list, test_list, label, test_label = handleData.DataReaderDouble(r"data/TinyData/", r"data/DiffPic/").read(default=True)
   
       # 2. prepare train data
       dataset = handleData.CustomedDataSet(train_x=pic_list, train_y=label, transform=globalParam.pic_transfer, target_transform=globalParam.label_transfer)
       dataloader = DataLoader(dataset, batch_size=21, shuffle=False, sampler=None,
                  batch_sampler=None, num_workers=0, collate_fn=None,
                  pin_memory=False, drop_last=False, timeout=0,
                  worker_init_fn=None)
   
       # 2. prepare test data
       testset = handleData.CustomedDataSet(train=False, test_x=test_list, test_y=test_label, transform=globalParam.pic_transfer, target_transform=globalParam.label_transfer)
       testloader = DataLoader(testset, batch_size=1, shuffle=False, sampler=None,
                  batch_sampler=None, num_workers=0, collate_fn=None,
                  pin_memory=False, drop_last=False, timeout=0,
                  worker_init_fn=None)
   
       # prepare model
       cnn = globalParam.bottleneckcnn_doubleinput_network
   
       # train
       print('--------------------------------------------')
       print('start training!')
       action.train(cnn, dataloader, testloader, globalParam.bottleneckcnn_doubleinput_optimizer, epoch=21)
   
       # test
       print('--------------------------------------------')
       print('start testing!')
       action.test_regression(cnn, testloader, draw_or_not=True)
   
       # # save，不要轻易save，容易冲掉以前的模型！
       # print('--------------------------------------------')
       # print('saving model!')
       # action.save_model(cnn, 'model/ResidualCNN/neck_batch=21_epoch=100_lr=0.00002_momentum=0.9_weight_decay=1e-2')
       #
       # # playback
       # print('--------------------------------------------')
       # print('playback model!')
       # playback_cnn = model.BottleneckCNN()  # 这里记得改！
       # action.model_playback(playback_cnn, 'model/ResidualCNN/neck_batch=21_epoch=100_lr=0.00002_momentum=0.9_weight_decay=1e-2')
       # action.test_regression(playback_cnn, testloader, draw_or_not=True)
   ```

3. 检查`globalParam.py`中的网络及训练参数符合预期：

   ```python
   # 网络
   cnn_network = model.BigCNN().to(device)
   nn_network = model.NN().to(device)
   residualcnn_network = model.ResidualCNN().to(device)
   bottleneckcnn_network = model.BottleneckCNN().to(device)
   bottleneckcnn_doubleinput_network = model.BottleneckCNNDoubleInput().to(device)
   
   # 优化器
   # lr:学习率，不宜过大, momentum:优化器冲量
   # weight_decay: 权重衰减系数（加上这个就相当于对权重进行L2范数约束，这个参数就相当于λ）
   cnn_optimizer = torch.optim.SGD(cnn_network.parameters(), lr=0.00001, momentum=0.9, weight_decay=1e-2)
   nn_optimizer = torch.optim.SGD(nn_network.parameters(), lr=0.2)
   residualcnn_optimizer = torch.optim.SGD(residualcnn_network.parameters(), lr=0.00001, momentum=0.9, weight_decay=1e-2)
   bottleneckcnn_optimizer = torch.optim.SGD(bottleneckcnn_network.parameters(), lr=0.00002, momentum=0.9, weight_decay=1e-2)
   bottleneckcnn_doubleinput_optimizer = torch.optim.SGD(bottleneckcnn_doubleinput_network.parameters(), lr=0.00002, momentum=0.9, weight_decay=1e-2)
   ```

4. 跑一下！
