"""


现在开始讲解神经网络的搭建。
我们可以对着官方文档当中的api来进行开发。
我们主要用到torch中的nn模块
from torch import nn
torch.nn: -> nn: neural network. 神经网络。


    Containers: 容器, 或者说骨架, 我们就是要根据
    Convolution Layers: 卷积层
    Pooling layers: 池化层
    Padding Layers: padding
    Non-linear Activations (weighted sum, nonlinearity): 非线性激活
    Non-linear Activations (other) #
    Normalization Layers:
    Recurrent Layers
    Transformer Layers
    Linear Layers
    Dropout Layers
    Sparse Layers
    Distance Functions
    Loss Functions
    Vision Layers
    Shuffle Layers
    DataParallel Layers (multi-GPU, distributed)
    Utilities
    Quantized Functions
    Lazy Modules Initialization


torch.nn.Containers:
这节课程，我们主要学习nn中的Containers。 也就是容器。
    实际上就是骨架了. 我们这节课就来学习这个Containers的配置.
    Containers: 包含的东西:
        Module          Base class for all neural network modules. -> 非常重要的
        Sequential      A sequential container.
        ModuleList      Holds submodules in a list.
        ModuleDict      Holds submodules in a dictionary.
        ParameterList   Holds parameters in a list.
        ParameterDict   Holds parameters in a dictionary.
示例:
Module:

Base class for all neural network modules. 是所有的神经网络模块的基础.
Your models should also subclass this class. 你的模型应该继承这个类.
Modules can also contain other Modules, allowing to nest them in a tree structure. You can assign the submodules as regular attributes:


    import torch.nn as nn
    import torch.nn.functional as F

    class Model(nn.Module): # 必须继承父类nn.Module
        def __init__(self):
            super().__init__() # 使用父类的初始化方法, 进行初始化
            # 定义自己的网络模型架构:
            self.conv1 = nn.Conv2d(1, 20, 5)
            self.conv2 = nn.Conv2d(20, 20, 5)

        # Defines the computation performed at every call. # 每次调用时的计算
        # Should be overridden by all subclasses. # 所有的子类应该重写.
        # 其实底层就是实现了__call__()方法.
        def forward(self, x): # 进行前向传播, 就是我们一层一层的要干什么.
            x = F.relu(self.conv1(x))
            return F.relu(self.conv2(x))
            # 返回的就是最终的网络输出结果.
            # forward中写的就是我们的神经网络架构.

"""
import torch
from torch import nn



# 创建一个块的方式. 
class TestModule(nn.Module): # 继承父类的Module
    """我们可以直接alt+insert进行快速重写父类方法."""
    
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input):
        output = input + 1
        return output
                
    

class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        # 定义需要用到的网络

    def forward(self, input):
        pass



def test01() -> None:
    test_module = TestModule()
    x = torch.tensor(1.0) # generate a scalar

    output = test_module(x) # 这就是将输入放入到神经网络当中, 默认执行forward这个函数

    print(output) # tensor(2.)








if __name__ == '__main__':
    test01()












































