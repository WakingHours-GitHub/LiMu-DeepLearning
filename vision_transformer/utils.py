import torch
import torchvision
from torchvision import transforms
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
from typing import List
import os
import cv2 as cv
import random
import math
from torch.nn import functional as F

def try_all_GPUS(idx=None) -> List[torch.device]:
    devices = []
    devices.extend([torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())])
    return devices if devices else "cpu"

def accuracy(y_hat:torch.Tensor, label:torch.Tensor):
    # because we don't use softmat layer, so we must add softmax operator in ouput.    
    return (F.softmax(y_hat, dim=1).argmax(dim=1) == label).float().mean() # this place is sum(), so we need use BS in metric. 


def train_cos(
    net: nn.Module, loss_fn: nn.Module,
    train_iter:DataLoader, test_iter:DataLoader,
    lr, num_epoch,
    momentum=0.937, weight_decay=5e-4,
    load_path:str=None, devices=try_all_GPUS(),
):
    eps = 0.35
    net = nn.DataParallel(net, devices).to(devices[0])
    loss_fn.to(devices[0])
    if load_path:
        print("load net parameters: ", load_path)
        net.load_state_dict(torch.load(load_path)) # load parameters for net module. 
    # trainer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    trainer = torch.optim.Adam(net.parameters(), lr=lr)

    lf = lambda x: ((1 + math.cos(x * math.pi / num_epoch)) / 2) * (1 - eps) + eps 
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer=trainer, # 优化器.
        lr_lambda=lf, # 学习率函数. 
    ) # 根据lambda自定义学习率. 

    metric = Accumulator(3)
    trainer.zero_grad() # empty.
    # train:
    print("train on:", devices)
    for epoch in range(num_epoch):
        net.train()
        metric.reset() # 重置
        # train a epoch. 
        for i, (x, labels) in enumerate(train_iter):
            x, labels = x.to(devices[0]), labels.to(devices[0])
            y_hat = net(x)
            loss = loss_fn(y_hat, labels)

            trainer.zero_grad() # first empty gradient
            loss.sum().backward() # than calculate graient by backwoard (pro)
            trainer.step() # update weight. 
        
            metric.add(loss.sum(), accuracy(y_hat, labels), 1)
        
        scheduler.step() # we only update scheduler. 

        # evaluate
        if (epoch+1) % 10 == 0:
            test_accuracy = evaluate_test_with_GPUS(net, test_iter)
            print(epoch+1, "test acc:", test_accuracy, "train loss:", metric[0]/metric[-1], "train acc:", metric[1]/metric[-1])

            try:
                torch.save(net.state_dict(), f"./logs/epoch{epoch+1}_testacc{test_accuracy:4.3}_loss{metric[0]/metric[-1]:3.2}_acc{metric[1]/metric[-1]:.2}.pth")
            except:
                os.mkdir("./logs")
                torch.save(net.state_dict(), f"./logs/epoch{epoch+1}_testacc{test_accuracy:4.3}_loss{metric[0]/metric[-1]:3.2}_acc{metric[1]/metric[-1]:.2}.pth")







def train(net: nn.Module, loss_fn, train_iter, vaild_iter, lr, num_epochs, start_point, 
        lr_period, lr_decay, weight_decay=5e-4,  momentum =0.9 ,devices=try_all_GPUS(), load_path:str = None):
    net = torch.nn.DataParallel(net, devices).to(device=devices[0])
    loss_fn.to(devices[0])
    if load_path:
        print("load net parameters: ", load_path)
        net.load_state_dict(torch.load(load_path)) # load参数. 

    trainer = torch.optim.SGD(net.parameters(), lr, momentum=0.9, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(trainer, step_size=lr_period, gamma=lr_decay) # 学习率衰减.
    # 每隔lr_period, 将当前lr乘以lr_decay, 达到学习率衰减的效果.

    metric = Accumulator(3)
    # 开始训练
    print("train on: ", devices)
    for epoch in range(num_epochs):
        net.train()
        metric.reset() # reset Accumulator. 
        
        for i, (features, labels) in enumerate(train_iter):
            features, labels = features.to(devices[0]), labels.to(devices[0])
            y = net(features)
            loss = loss_fn(y, labels)
            
            trainer.zero_grad()
            loss.sum().backward()
            trainer.step()

            metric.add(loss.item(), accuracy(y, labels), 1) # 因为都是使用的平均值, 所以这里加1
            # 如果loss, 和accuracy使用的是sum. 那么这里就要改成labels.shape[0]也就是多少个批量.
        if epoch > start_point: # 大于之后我们才开始衰减。
            scheduler.step() # 每轮结束后我们要更新一下这个scheduler. 
            # print(scheduler.get_lr())
            


        # save net paramters: 
        if (epoch+1) % 10 == 0:
            test_accuracy = evaluate_test_with_GPUS(net, vaild_iter)
            print(epoch+1, "test acc:", test_accuracy, "train loss:", metric[0]/metric[-1], "train acc:", metric[1]/metric[-1])

            try:
                torch.save(net.state_dict(), f"./logs/epoch{epoch+1}_testacc{test_accuracy:4.3}_loss{metric[0]/metric[-1]:3.2}_acc{metric[1]/metric[-1]:.2}.pth")
            except:
                os.mkdir("./logs")
                torch.save(net.state_dict(), f"./logs/epoch{epoch+1}_testacc{test_accuracy:4.3}_loss{metric[0]/metric[-1]:3.2}_acc{metric[1]/metric[-1]:.2}.pth")
        

def evaluate_test_with_GPUS(net: nn.Module, test_iter):
    if isinstance(net, nn.Module):
        net.eval() # set module on evaluation mode

    device = next(iter(net.parameters())).device # 拿出来一个参数的device
    matrix = Accumulator(2)
    with torch.no_grad():
        for X,  labels in test_iter:
            X, labels = X.to(device), labels.to(device)
            # matrix.add(accuracy(net(X), label), label.shape[0])
            # print(net(X).shape)
            
            matrix.add(accuracy(net(X), labels), 1) # here, accuracy used 'mean()' so, add 1.
    return matrix[0] / matrix[1] # 均值.
    


path_join = lambda *args: os.path.join(*args)


class CIFAR10_dataset(Dataset):
    def __init__(self, type_dataset: str = "train", vaild_rate=0.1) -> None:
        super().__init__()
        self.type_dataset = type_dataset
        self.vaild_rate = vaild_rate
        self.transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(42),
            transforms.RandomResizedCrop(32, (0.6, 1.0), ratio=(1.0, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize([0.4914, 0.4822, 0.4465],  # normalize. 归一化.
                                 [0.2023, 0.1994, 0.2010])
        ])
        self.transform_vaild = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.4914, 0.4822, 0.4465],  # normalize. 归一化.
                                         [0.2023, 0.1994, 0.2010])
                ])
        self.labels_dict = self.parse_csv2label()
        self.classes = ['airplane', 'automobile', 'bird', 'cat',
                        'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

        self.root_path = path_join("../kaggle/cifar10/", "train")
        if "train" == self.type_dataset:
            # generate train and vaild dataset file.
            self.shuffle_train_vaild()
            # only generate in "train" mode.

            with open("../kaggle/cifar10/train.txt", "r") as f:
                self.file_path_list = f.readlines()

        elif "vaild" == self.type_dataset:
            with open("../kaggle/cifar10/train_vaild.txt", "r") as f:
                self.file_path_list = f.readlines()
        else: # test:
            self.root_path = path_join("../kaggle/cifar10/", "test")

            self.file_path_list = [path_join(self.root_path, file_name) for file_name in os.listdir(self.root_path)]


    def __getitem__(self, index):
        file_path = self.file_path_list[index].strip()
        file_name = file_path.split("/")[-1].split(".")[0]
        img = cv.imread(file_path)
        if "train" == self.type_dataset:
            X = self.transform_train(img)
            return X, self.classes.index(self.labels_dict[file_name])
        elif "vaild" == self.type_dataset:
            X = self.transform_vaild(img)

            return X, self.classes.index(self.labels_dict[file_name])

        else: # test
            X = self.transform_vaild(img)
            return X, file_name

        


    def parse_csv2label(self):
        with open("../kaggle/cifar10/trainLabels.csv", "r") as f:
            return {ele[0]: ele[1] for ele in [line.strip().split(',') for line in f.readlines()][1:]}

    def shuffle_train_vaild(self):
        l = len(os.listdir(self.root_path))

        try:
            file_name_list = os.listdir(self.root_path)
            random.shuffle(file_name_list)

            with open(path_join(self.root_path, "../", "train.txt"), "w") as train_file_writer:
                train_file_writer.write(
                    "\n".join([path_join(self.root_path,  file_name)
                              for file_name in file_name_list[0: int(l*(1-self.vaild_rate))]])
                )

            with open(path_join(self.root_path, "../",  "train_vaild.txt"), "w") as train_file_writer:
                train_file_writer.write(
                    "\n".join([path_join(self.root_path,  file_name)
                              for file_name in file_name_list[int(l*(1-self.vaild_rate)):]])
                )

        except Exception as e:
            print("error: ", e)

    def __len__(self):
        return len(self.file_path_list)




def load_CIFAR10_iter(batch_size=64, num_workers=28):
    return DataLoader(
        CIFAR10_dataset("train"),
        batch_size,
        shuffle=True,
        num_workers=num_workers
    ),     DataLoader(
        CIFAR10_dataset("vaild"),
        batch_size,
        shuffle=True,
        num_workers=num_workers
    )



class Accumulator():
    def __init__(self, n) -> None:
        self.data = [0.0]*n # 初始化

    def add(self, *args):
        self.data = [a+float(b) for a, b in zip(self.data, args)]
    
    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx] 



def train_cifar10(net, lr, batch_size, num_epoch):
    print("train!!!")
    devices = try_all_GPUS()
    # train_iter, vaild_iter = load_data_CIFAR10(batch_size)
    train_iter, vaild_iter = load_CIFAR10_iter(batch_size)

    # net = nn.DataParallel(net, devices).to(devices[0])
    net = net.to(devices[0])

    loss_fn = nn.CrossEntropyLoss().to(devices[0])
    trainer = optim.SGD(net.parameters(), lr, weight_decay=1e-4)

    # if isinstance(net, nn.Module):
        # net.train()
    metric = Accumulator(3)

    for epoch in range(num_epoch):
        net.train()
        metric.reset()
        for X, label in train_iter:
            X, label = X.to(devices[0]), label.to(devices[0])
            y_hat = net(X)

            trainer.zero_grad()
            loss = loss_fn(y_hat, label)
            loss.sum().backward()
            trainer.step()

            metric.add(accuracy(y_hat, label), loss.item(), 1)
        
        if (epoch+1) % 10 == 0:
            try:
                torch.save(net.state_dict(), f"./logs/epoch{epoch+1}_testacc{evaluate_test_with_GPUS(net, vaild_iter):3.2}_loss{metric[1]/metric[-1]:3.2}_acc{metric[0]/metric[-1]:.2}.pth")
            except:
                os.mkdir("./logs")
                torch.save(net.state_dict(), f"./logs/epoch{epoch+1}_testacc{evaluate_test_with_GPUS(net, vaild_iter):3.2}_loss{metric[1]/metric[-1]:3.2}_acc{metric[0]/metric[-1]:.2}.pth")


        print(epoch, loss.item(), metric[0]/metric[-1])

