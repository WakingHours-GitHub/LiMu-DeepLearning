"""
Normalization
正则化层. 可以优化网络.

Recurrent Layers
一些写好的网络结构,

Transformer Layers
也是一种写好的网络结构

Linear Layers
线性层

Dropout Layers
    nn.Dropout()
    就是在训练的过程中, 随机丢弃一些其中的因素,
    按照p(概率)去丢失. 为了防止过拟合现象
Distance Layers
    就是计算误差
Loss Function:
    就是损失函数.


这里主要看线性层:
    其实也就是全连接层 FULL_
torch.nn.Linear(in_features, out_features, bias=True, device=None, dtype=None)
    torch封装的非常好, 只要输入, 和输出以及是否要偏置就可以实现
实际上就是矩阵相乘, 使得最后的结果得到我们想要的形状
因为是矩阵相乘, 所以输入的tensor必须是一个二维的
所以一般再使用Linear之前, 我们一般会使用reshape和view, 这是手动将结果处理成二维.
我们也可以使用flatten. 作用是: 除了第一维度, 将后面的所有维度进行展平.
torch.flatten. 或者是nn.Flatten
所以一般是: [barch, ...]然后与矩阵相乘, 根据我们的要求,  得到最终的结果
input:(*, input)
output: (*, output)


in_features – size of each input sample
out_features – size of each output sample
bias – If set to False, the layer will not learn an additive bias. Default: True



# torchvision中, 已经提供了非常全面的网络结构. 我们可以直接调用这里面的一些网络模型. 
torchvision.models
这里面提供了一些非常经典对于图像方面的网络结构.
别人已经训练好的模型, 你拿过来直接使用就好了.

 """
import torch
from torch.utils.data import DataLoader, dataloader
import torchvision
from torch import nn

BATCH_SIZE = 64


class MyLinear(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.flatten = nn.Flatten()  # 除了dim=1. 之外将其他dim平坦.
        self.linear = nn.Linear(3072, 10) # 实际上就是做矩阵乘法。 

    

    def forward(self, X):
        # X = self.flatten(X)
        X = X.reshape((BATCH_SIZE, -1)) # 直接reshape也可以
        X = self.linear(X)
        return X





        

def linear_test() -> None:
    CIFAR10_dataset = torchvision.datasets.CIFAR10(
        "../data",
        train=False,
        transform=torchvision.transforms.ToTensor(),
        download=True,
    )
    CIFAR10_dataloader = DataLoader(
        CIFAR10_dataset, BATCH_SIZE, 
    )

    for idx, (images, labels) in enumerate(CIFAR10_dataloader):
        print(images.shape) # torch.Size([64, 3, 32, 32])
        

        output = MyLinear()(images)

        print(output.shape) 
        # 经过flatten layer: torch.Size([64, 3072])
        # torch.Size([64, 10])

        break


    










if __name__ == "__main__":
    linear_test()











 