"""
损失函数
作用:
1. 衡量我们输出和目标之间的一个差距
2. 用于反向传播(为我们的更新提供一定依据)

loss_function在torch.nn中. 
我们可以通过查阅官方提供的API手册来知道如何使用这些提供的loss函数,
我们需要注意的是各个参数的输入形状.
 





"""

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms 


# 先来定义一下我们的输入和label案例吧:
inputs = torch.tensor([1.0, 2, 3])
labels = torch.tensor([1.0, 2, 5])

# 然后我们需要reshape成4D的shape
input = inputs.reshape(shape=(1, 1, 1, 3))
labels = labels.reshape(shape=(1, 1, 1, 3))


# 我们自己的网络模型: 

class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        # 输入图像:
        # x: [batch_size, 3, 32, 32]
        self.conv1=nn.Conv2d(
            in_channels=3, # 输入的通道数
            out_channels=32, # 输出的通道数
            kernel_size=(5, 5),
            stride=(1, ),
            padding=(2, ),
        )
        # 输出: x[batch_size, 32, 32, 32]
        # 卷积层输出公式:
        # H_out = (H-F+2P)/s + 1
        #

        # x: [batch_size, 32, 32, 32]
        self.maxpool1=nn.MaxPool2d(
            kernel_size=2
        )
        # x: [batch_size, 32, 16, 16]
        self.conv2=nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=(5, 5),
            stride=(1, ),
            padding=(2, )
        )
        # x: [batch_size, 32, 16, 16]
        self.maxpool2=nn.MaxPool2d(
            kernel_size=2,
        )
        # x: [batch_size, 32, 8, 8]

        self.conv3=nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=(5, 5),
            stride=(1, ),
            padding=(2, )
        )
        # x[batch_size, 64, 8, 8]
        self.maxpool3=nn.MaxPool2d(
            kernel_size=2
        )
        # x: [batch_size, 64, 4, 4]

        # 目前为止图像是: x: [batch_size, 64, 4, 4]
        self.flatten = nn.Flatten() # 和torch.flotten()一样
        # 只对dim=1(包括)以后的维度进行平坦.

        # 到这里时, 输入图像已经是:
        # 线性层也称之为全连接层.
        # 定义的就是一个矩阵的行和列, in就是行, out就是列
        # 也就是这样的一个矩阵: [in, out] # 然后这一层要左的就是矩阵相乘
        # 1024 = 64*4*4
        self.linear1=nn.Linear(
            in_features=64*4*4,
            out_features=64
        )
        # 经过线性层,
        # 为64
        self.linear2=nn.Linear(
            in_features=64,
            out_features=10 # 最后分成10个类别.
        )

    # 在构建网络的时候, 我们需要时时刻刻的关注我们x形状的一个变化
        self.model1 = torch.nn.Sequential(

        )


    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.maxpool3(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.linear2(x)
        return x



def test04() -> None:
    """
    自动计算梯度, pytorch中有自动计算梯度的模块. 我们可以直接使用

    """
    CIFAR10_dataset = datasets.CIFAR10("./data/", train=False, transform=transforms.ToTensor(), download=True)
    CIFAR10_dataloader = DataLoader(CIFAR10_dataset, 32, True, num_workers=4)

    net = MyModule() # 创建网络实例. 
    loss = nn.CrossEntropyLoss()
    
    for idx, (images, labels) in enumerate(CIFAR10_dataloader):
        y_hat = net(images)
        # 打印一下输入和输出形状, 决定我们用什么损失函数
        print(y_hat.shape)
        print(labels.shape)
        # torch.Size([32, 10])
        # torch.Size([32]) # 所以可见, 我们是可以进行crossEntropy的损失的. 
        loss_value = loss(y_hat, labels)
        print(loss_value) # tensor(2.3422, grad_fn=<NllLossBackward0>) # 得到的误差

        # 反向传播. 
        loss_value.backward() # 反向传播. 
        # 经过backward, 我们就可以计算出每层中的参数的梯度.
        # 然后我们的优化器就会利用这些梯度进行优化我们的参数.



        break



def test03():
    """
    交叉熵损失。
    交叉熵损失常常用到多分类问题中.
    公式就是: -log( (y_pred.*label)/sum(exp(y_pred) ) # 只关注正确类别的置信度, 使其最大即可. 
    label应该是一个one_hot编码, 但是, 这里只需要传入label真实值对应的位置就可以了
    注意形状: C是类别. 
    input: (C), (N, C), (N, C, ...) 计算K维度的loss
    target: (), (N), (N, ...)
    # 注意, 输入一般是(barch_size, 多分类的类别)
    # 而target只有(batch_size) 然后每个数, 对应的就是真实的类别, (的编码)
    # 返回的是scalar就是一个标量张量.

    -log(exp(y_pred[label])/sum(exp(pred)))
    
    :return:
    """
    x = torch.tensor([0.1, 0.2, 0.3]) # 按道理, 我们应该先将x进入softmax进行运算.
    x = nn.Softmax()(x) # 经过softmax算子. 
    print(x) # tensor([0.3006, 0.3322, 0.3672]) # 得到我们的输出概率.
    x = x.reshape(shape=(1,  3))  # 形状: [batch, Classes] # 分别是batch和类别.
    y = torch.tensor([1]) 

    # 所以, 可见, 我们应该是
    
    # 然后再经过CrossEntropyLoss
    loss = nn.CrossEntropyLoss()
    result = loss(x, y)

    print(result) # tensor(1.1001)
 



def test02() -> None:
    """
    计算MSELoss, 记住, 是MSE, Mean Squared Error
    也就是均方误差
    :return:
    """
    result = nn.MSELoss()(inputs, labels)
    print(result) # tensor(1.3333)


def test01() -> None:
    """
    nn.L1Loss(reduction="sum")(pred, label)
    # 这就计算pred和label之间的差值的绝对值
    :return:
    """
    result_avg = nn.L1Loss()(input, labels)
    print(result_avg.item()) # tensor(0.6667) # 0.6666666865348816
    # 如果是单个元素的情况下, 我们可以使用item直接提取数据.

    result_sum = nn.L1Loss(reduction="sum")(inputs, labels) # 可以选择计算方式. 
    print(result_sum.item()) # 2.0
 


if __name__ == "__main__":
    # test01()
    # test02()
    # test03()
    test04()
    
    
    