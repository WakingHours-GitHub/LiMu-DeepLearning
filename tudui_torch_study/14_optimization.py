"""
优化器. 
优化器就是利用loss.backward得到的梯度, 进行更新参数, 从而使我们的loss不断减小.

优化器都在torch.optim这个包中.
我们可以查看官方文档, 来看看优化器是如何使用的. 

步骤:
    1. 构造一个优化器
        一般都是: optim_obj = optim.*(模型的参数(), 学习率, 以及一些特定优化器中特有的参数)
        例子: 
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        optimizer = optim.Adam([var1, var2], lr=0.0001)
        lr: 学习率.
    2. 调用optim_obj.step方法
        利用loss.backward得到的grad, 对传入进去的参数进行更新. 
    3. 

# 例子: 
for input, target in dataset:
    optimizer.zero_grad() # 清空梯度
    output = model(input) # 利用创建好的模型进行计算
    loss = loss_fn(output, target) # 计算loss值
    loss.backward() # 得到model中每个变量的一个梯度
    optimizer.step() # 根据上面得到的梯度, 对每一个变量进行更新


"""

# 现在让我们利用优化器直接搭建一个比较完整的神经网络吧.
import torch
from torch import nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import os
import sys
runtime_path = sys.path[0] # /home/wakinghours/programming/LiMu-DeepLearning/tudui_torch_study
# 能够获取到文件目录所在的文件夹
# os.getcwd() # 只能获取工作区的文件夹. 


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu" )
print(device)


class NetWork(nn.Module):
    def __init__(self):
        super(NetWork, self).__init__()
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
        
        # 我们还需要加上一个softmax层进行概率的映射
        x = nn.Softmax(dim=-1)(x)
        return x

def calculate_accuracy(y_hat, labels):
    return (torch.argmax(y_hat, dim=-1) == labels).float().mean()

def main(is_load=True) -> None:

    CIFAR10_dataset_train = torchvision.datasets.CIFAR10("./data", train=True, transform=transforms.ToTensor(), download=True)
    CIFAR10_dataloader_train = DataLoader(
        CIFAR10_dataset_train, 64, True, num_workers=8, 
        prefetch_factor=2, # 提前加载的sample数量
        )

    CIFAR10_dataset_test = torchvision.datasets.CIFAR10("./data", train=False, transform=transforms.ToTensor(), download=True)
    CIFAR10_dataloader_test = DataLoader(
        CIFAR10_dataset_test, 64, True, num_workers=8, 
        prefetch_factor=2, # 提前加载的sample数量
        )
    # generate network
    net = NetWork() # generate netword object
    if is_load:
        net.load_state_dict(torch.load(os.path.join(runtime_path, "14_net_dict.net")))
    net.to(device)
    print(iter(net.parameters()).__next__().device) # 看一下第一个参数的设备在哪, 检测一下我们是否已经搬到GPU上面了

    # generate loss function:
    # beacuse of is a classification problems
    loss = nn.CrossEntropyLoss() # definited Cross Entropy Loss function. 
    loss.to(device)

    # definited optimer:
    optim = torch.optim.SGD(
        params=net.parameters(), # 放所有网络中的参数.
        lr=0.01 # 学习率, 控制步长
    )

    for epoth in range(100):
        
        net.train() # start train mode
        for idx, (images, labels) in enumerate(CIFAR10_dataloader_train):
            images = images.cuda(device)
            labels = labels.cuda(device)
            
            y_hat = net(images)

            net.zero_grad()
            current_loss = loss(y_hat, labels)

            current_loss.backward()

            optim.step()


        # 每一轮保存一下: 
        torch.save(net.state_dict(), runtime_path+"/14_net_dict.net")
        

        # 在测试机当中进行测试: 
        net.eval()
        test_images, labels_test = iter(CIFAR10_dataloader_test).__next__() #  应该还需要判断一下是否到了结尾, 否则iter取不出来. 就报错了啊. 
        
        test_images = test_images.cuda(device)
        labels_test = labels_test.cuda(device)
        # RuntimeError: Input type (torch.FloatTensor) and weight type (torch.cuda.FloatTensor) should be the same or input should be a MKLDNN tensor and weight is a dense tensor
        # 报错, 在不是在同一个设备当中
        y_hat_test = net(test_images)
        current_loss = loss(y_hat_test, labels_test)
        
        accuracy = calculate_accuracy(y_hat_test, labels_test)
        print(f"epoch: {epoth}, loss: {current_loss.item()}, accuracy: {accuracy.item()} ", )
    



    
    




if __name__ == "__main__":
    main()
