"""
第一种网络训练的模型:
    模型, 数据, 损失函数
    .cuda(device: int) 即可将数据挪动到GPU上. device表示GPU的标号.

google colab, 提供了免费的GPU供我们使用. 基于Jupyter Notebook.
修改: 笔记本设置: 可以选择CPU, GPU, TPU(TPU就是Google自己研发的芯片, Tensor Processing Unit)

第二种网络训练模型:
    模型, 数据, 损失函数. 
    .to(ddevice)



"""

import time
import os
import torch
import torchvision
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import sys
from torch import nn
from tqdm import tqdm, trange

BATCH_SIZE = 64
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
EPOCH = 60



def concat_with_runtime_path(*args):
    runtime_path = sys.path[0]
    return os.path.join(runtime_path, *args)
    
writer = SummaryWriter(concat_with_runtime_path("./17_file"))
    

# print(concat_with_runtime_path("../", "./data"))

# 准备数据集：
train_data = torchvision.datasets.CIFAR10(
    root=concat_with_runtime_path("../", "./data"),
    train=True,
    transform=transforms.ToTensor(),
    download=True
)
test_data = torchvision.datasets.CIFAR10(
    root=concat_with_runtime_path("../", "./data"),
    train=False,
    transform=transforms.ToTensor(),
    download=True
)
# 利用dataloade进行加载:
train_dataloader = DataLoader(
    train_data,
    BATCH_SIZE,
    shuffle=True,
    num_workers=8,
)
test_dataloader = DataLoader(
    test_data,
    BATCH_SIZE,
    shuffle=True,
    num_workers=8,
)


# 搭建神经网络:
class CIFAR10_net_baed_VGG(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(
            self.get_vgg_block(3, 32, 1),
            self.get_vgg_block(32, 32, 1),
            self.get_vgg_block(32, 64, 1),
            nn.Flatten(),
            nn.Linear(64*4*4, 64),
            nn.Linear(64, 10), # 最后我们其实是不需要加上softmax的, 因为在做CrossEntropyError时就做了softmax

        )

    def get_vgg_block(self, in_channel: int, out_channel: int, num_conv: int) -> nn.Sequential:
        assert isinstance(num_conv, int)
        block = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, 1, 1)
        )
        for _ in range(num_conv-1):
            block.add_module("Conv2d", nn.Conv2d(out_channel, out_channel, 3, 1, 1))
        block.add_module("MaxPool", nn.MaxPool2d(kernel_size=2)) # torch中, MaxPooling的stride与K一样. 

        return block

    def get_net_construction(self) -> None:
        print(self)

    def get_net_layer_shape(self, input):
        print(f"source shape:{input.shape}")
        for model in self.model:
            input = model(input)
            print(f"{model}: {input.shape}")

    def forward(self, input) -> torch.Tensor:
        return self.model(input)
            
"""
每一个特定的问题都有一个特定的方式去评价. 
那么在分类问题中, 实际中评价模型的好坏, 主要是依靠accuracy, 也就是准确率
"""
# 分类问题中的正确率问题: 
def calculate_accuracy(y_hat, labels) -> float:
    # print(torch.nn.functional.softmax(y_hat, dim=-1).shape)
    y_hat = torch.argmax(torch.nn.functional.softmax(y_hat, dim=-1)) # 对最后一维做softmax, 然后按照最后行做

    return (y_hat == labels).float().mean() # 逻辑矩阵.float().mean() # 
    



def train():
    net = CIFAR10_net_baed_VGG()
    # net.to(device)
    net = net.cuda()
    loss_fu =nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    # 优化器没有.cuda()方法

    for epoch in trange(EPOCH, desc="train"):
        net.train() # 开启训练模式
        """
        set the module in training mode.
        train只有对某些特定的层起作用, 例如Dropout, BatchNorm, etc层有作用, 
        dropout只有在训练的时候起作用, 推理的时候我们希望模型稳定一点, 不要出现随即性, 所以推理的时候通常关闭dropout.
        BatchNorm, 是批量归一化, 也是解决训练深层网络时出现数据不稳定的问题. 就是线性变换到一个学习到的分布中.
        """
        


        # batch
        for batch_index, (inputs, labels) in enumerate(train_dataloader):
            # 数据: .cuda()
            inputs, labels = inputs.cuda(), labels.cuda()
            y_hat = net(inputs)

            loss = loss_fu(y_hat, labels)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            writer.add_scalar("train_loss", loss.item(), batch_index)
    

        net.eval() # 开启评估. 
        """
        set the module in evaluation mode.
        也是对几个特定的层起作用. 
        如果有这几个层, 我们要开启评估模型, 或者训练模式, 确保我们的模型正确. 
        """
        # 评估有没有训练好: 
        # 对测试集进行测试: 不需要调优, 所以直接是no_grad()区域.
        with torch.no_grad():
            test_mean_error = 0
            test_mean_accuracy = 0 # 测试集平均测试.
            # inputs, labels = iter(test_dataloader).__next__() # 只看一个batch的test的精度。 
            for inputs, labels in test_dataloader:
                inputs, labels = inputs.cuda(), labels.cuda()

                y_hat = net(inputs)
                
                loss = loss_fu(y_hat, labels)

                test_mean_error += loss.item() # 
                test_mean_accuracy += calculate_accuracy(y_hat, labels)
            test_mean_error /= len(test_dataloader) # 求平均。 


            os.system("clear") # 清空控制台. 
            print(f"epoch: {epoch}, test total loss: {test_mean_error}, test mean accuracy:{test_mean_accuracy}")
            # item(), 就是无论数据多少层, 只要最终的结果是一个数据, 我们直接取出该数据. 
            
            # 画图
            writer.add_scalar("test_loss", test_mean_error, epoch)


        
        # 保存模型: 只保存字典. 
        torch.save(net.state_dict(), concat_with_runtime_path(f"./17_file/CIFAR10_epoch_{epoch}_test_{test_mean_error}.pth"))

        



if __name__ == '__main__':
    train()