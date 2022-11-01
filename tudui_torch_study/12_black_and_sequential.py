"""
nn.Sequtial(*args): 序列. 我们可以理解成为是一个容器, 其中参数类型都是nn.Module类型.

本质上就是nn.Module的东西, 其中我们还具体实现过. 这里不在赘述. 
Sequtial()就是实现一个简单的网络模型, 其中不含有任何控制流. 


所以我们利用Sequential来简单实现一下对CIFAR10进行的简单分类. 
这里我们选择经典的网络模型进行搭建神经网络.







"""

import torch
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torch import nn 
nn.CrossEntropyLoss()

class MyNN(nn.Module):









