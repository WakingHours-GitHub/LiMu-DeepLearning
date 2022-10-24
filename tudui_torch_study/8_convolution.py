import torchvision
from torchvision import transforms
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import nn
import torch.nn.functional as F # 这里面是一些常用的函数.
import cv2 as cv
import numpy as np

write = SummaryWriter("./logs")

"""
首先我们来学习卷积这个操作. 
我们首先学习一下torc.nn.functional.conv2d(), 也就是这个卷积操作.


然后使用更高级的API: 在torch.nn中. 我们可以导入
"""

# 首先实现我们自己的Module:
class MyConvolution2Dimension(nn.Module):
    def __init__(self):
        super(MyConvolution2Dimension, self).__init__()
        self.conv2d = nn.Conv2d(
            in_channels=3,  # 输入通道是3
            out_channels=6, # 输出通道是6, 也就是我们使用六个卷积核进行扫描
            kernel_size=(3, 3), # 可以传入int和tuple
            stride=1,
            padding=0,
            bias=True, # 每个卷积核都使用一个偏执.
        )


    def forward(self, X):
        return self.conv2d(X)


def conv2d_layer() -> None:
    """
    conv2d的实战, 我们首先自己定义自己的网络模型, 然后生成数据集
    """
    CIFAR10_datasets = torchvision.datasets.CIFAR10("./torch_datasets", False, transform=transforms.ToTensor(), download=True)
    # 生成DataLoader
    CIFAR10_dataloader = DataLoader(CIFAR10_datasets, 64, True, num_workers=4) # 使用四个进程去读入。


    # 然后我们就可以遍历这个. datalloader
    for idx, (feature, label) in enumerate(CIFAR10_dataloader):
        print("feature.shape", feature.shape)
        # feature.shape torch.Size([64, 3, 32, 32])

        # 生成网络实例, 直接调用call函数, 然后直接将feature传入进去, 做forward操作.
        y_hat = MyConvolution2Dimension()(feature)
        # 经过卷积层处理:
        print("y_hat.shape", y_hat.shape)
        # y_hat.shape torch.Size([64, 6, 30, 30])
        # 可见, 经过卷积层变换后, 可以看到, channel变成了6, 并且长宽都发生变化了
        # 计算一下(32-3+2*0)/1 + 1 -> 30.

        # 尝试写入到TensorBoard.
        # 不过我们在向TensorBoard add_images时, 我们要注意channel必须是3D的. 不过为了看看我们卷积层输出的结果
        # 我们这里做一个不那么严谨的操作: 也就是reshape一下
        write.add_images("before_conv2d:", feature, idx)
        y_hat = y_hat.reshape((-1, 3, 30, 30)) # create view
        write.add_images("after_conv2d", y_hat, idx)

        #


        break # 看一个batch即可.
 #






    





def test_torch_nn_functional_conv2d() -> None:
    input = torch.tensor([ # 手动创建一个输入.
        [1, 2, 0, 3, 1],
        [0, 1, 2, 3, 1],
        [1, 2, 1, 0, 0],
        [5, 2, 3, 1, 1],
        [2, 1, 0, 1, 1]
    ]).reshape((1, 1, 5, 5)) # 我们要手动reshape一下,
    # 变成: [batch_size, channel, h, w)



    # 自定义卷积核:
    kernel = torch.tensor([
        [1, 2, 1],
        [0, 1, 0],
        [2, 1, 0]
    ]).reshape((1, 1, 3, 3))
    # [in_channel, out_channel, k_h, k_w]

    print("input.shape", input.shape) # input.shape torch.Size([5, 5])


    output_1 = F.conv2d(
        input=input, # 表示输入.
        weight=kernel, # 就表示卷积核
        bias=None, # 可以为每个卷积核创建对应的bias.
        stride=1, # 如果是int, 则横竖方向步长都是1.
        padding=0, # 也就是不填充周围区域.
    )
    print(output_1)
    # tensor([[[[10, 12, 12],
    #           [18, 16, 16],
    #           [13,  9,  3]]]])
    print(output_1.shape) # torch.Size([1, 1, 3, 3])




    output_2 = F.conv2d(input, kernel, None, 2, 0)
    # set stride is 2. then shape are 1, 1, 2, 2
    print(output_2)

    output_2 = F.conv2d(input, kernel, None, 1, 1)
    # set padding is 1. then shape are 1, 1, 2, 2
    print(output_2.shape) # torch.Size([1, 1, 5, 5])
    # tensor([[[[ 1,  3,  4, 10,  8],
    #           [ 5, 10, 12, 12,  6],
    #           [ 7, 18, 16, 16,  8],
    #           [11, 13,  9,  3,  4],
    #           [14, 13,  9,  7,  4]]]])

    # so 计算公式如下:
    # 计算公式:
    # new_H = [(H+2*padding-K)/stride + 1]向下取整
    # K是kernel的长度.

if __name__ == '__main__':
    # test_torch_nn_functional_conv2d()
    conv2d_layer()











write.close()