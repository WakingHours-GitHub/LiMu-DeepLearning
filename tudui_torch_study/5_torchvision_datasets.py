"""
通过查看官网API:
    torchvision, build in datasets.

    也就是torchvision中内建的数据集.

这节课我们来讲解torchvision中内建的数据集的使用.
    包名: torchvision.datasets.XXX
以及配合transforms.的使用.



torchvision中的数据集的使用:
    root (string) – Root directory of dataset where directory cifar-10-batches-py exists or will be saved to if download is set to True.
    train (bool, optional) – If True, creates dataset from training set, otherwise creates from test set.
    transform (callable, optional) – A function/transform that takes in an PIL image and returns a transformed version. E.g, transforms.RandomCrop
    target_transform (callable, optional) – A function/transform that takes in the target and transforms it.
    download (bool, optional) – If true, downloads the dataset from the internet and puts it in root directory. If dataset is already downloaded, it is not downloaded again.

    root: 数据集的位置. 当数据集不存在时, 默认下载到该地方.
    train: 是否是训练集.
    transform: 就是要对input(feature)进行的预处理. 与前面讲解的transform对应
    target_transform: 就是对label进行的预处理.
    download: 是否下载


    介绍一下CIFAR-10 dataset:
        该数据集共有60000张彩色图像，这些图像是32*32，分为10个类，每类6000张图。
        这里面有50000张用于训练，构成了5个训练批，每一批10000张图；另外10000用于测试，单独构成一批。
        测试批的数据里，取自10类中的每一类，每一类随机取1000张。抽剩下的就随机排列组成了训练批。
        注意一个训练批中的各类图像并不一定数量相同，总的来看训练批，每一类都有5000张图。

对于其他数据集, 我们只需要看torchvision中的官方文档我们就可以去使用该数据集了.
了解各个参数. 不过大同小异.
下载地址, 我们可以点击到torchvision.datasets.XXX然后显示即可.
然后找到其中的url连接, 就是下载地址. 


"""
import torchvision
from torchvision import transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch
import os, sys
from torch.utils.data import DataLoader


os.chdir(sys.path[0])


writer = SummaryWriter("./logs")




def combine_datasets_transform() -> None:
    CIFAR10_transform = transforms.Compose([
        transforms.ToTensor(),
    ]) 
    train_set = torchvision.datasets.CIFAR10( # 在dataset的时候就设置好transforms     
        "../data", 
        transform=CIFAR10_transform, # 这里设置上面定义好的transform. 这样我们就可以,
        # 将读取之后的image进行预处理, 经过transform.
        download=True

    )
    test_set = torchvision.datasets.CIFAR10(
        "../data",
        train=False,
        transform=CIFAR10_transform,
        download=True
    )

    for i in range(10):
        img, label = train_set[i] # 取出数据集中的样本。
        writer.add_image("combine of torchvision datasets and transform", img, i)




def torchvision_datasets() -> None:
    train_set = torchvision.datasets.CIFAR10(
        root="../data", # 数据保存的地址
        train=True, # 默认为true,
        transform=None,
        download=True
    )
    test_set = torchvision.datasets.CIFAR10("../data", True, None, download=True)

    # 这些数据集的样式: ((feature, label), ...)
    # 是一个双重嵌套迭代器. 第一层包含着每个样本, 然后每个样本中又包含着feature和label
    # 然后还有各种各样的属性, 通过debug我们可以查看其属性.
    # 例如classes: 就是该数据集的标签集合.
    # 查看:
    print(train_set[0]) # (<PIL.Image.Image image mode=RGB size=32x32 at 0x7F26BE7E9720>, 6)
    print(train_set.classes) # ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    img, target = test_set[0]
    print(img, target)
    print(train_set.classes[target])

def torchvision_datalaoder() -> None:
    trans = transforms.Compose([        
        transforms.Resize(40),
        transforms.ToTensor(),
        transforms.RandomCrop(32)
    ])
    train_set = torchvision.datasets.CIFAR10("../data", True, transform=trans, download=True)
    train_iter = DataLoader(train_set, 32, True, num_workers=16)
    
    imgs, labels = iter(train_iter).__next__()
    
    writer.add_images("images", imgs, 1)
    



if __name__ == '__main__':
    # torchvision_datasets()

    # combine_datasets_transform()
    
    
    torchvision_datalaoder()









writer.close() # 资源释放。
