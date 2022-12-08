import torch
import os
import random
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import cv2 as cv
import numpy as np
from torch import nn
from torch import optim
import pandas as pd

from typing import List


def try_all_gpus() -> List[torch.device]:
    devices = []
    devices.extend([torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())])
    return devices if devices else "CPU"

# print(try_all_gpus())


def train(net: nn.Module, loss, optim: optim, lr, weight_decay, num_eopchs, batch_size, train_iter, test_iter, devices = try_all_gpus()):
    pass







def evaluate_test_with_GPUS(net: nn.Module, test_iter):
    if isinstance(net, nn.Module):
        net.eval() # set module on evaluation mode

    device = next(iter(net.parameters())).device # 拿出来一个参数的device
    matrix = Accumulator(2)
    with torch.no_grad():
        for X,  label in test_iter:
            X, label = X.to(device), label.to(device)
            matrix.add(accuracy(net(X), label), label.shape[0])
    return matrix[0] / matrix[1]
    
        
        


def accuracy(y_hat:torch.Tensor, label:torch.Tensor):
    return (y_hat.argmax(dim=1) == label).float().mean()




def train_cifar10(net, lr, num_epochs, train_iter, test_iter, devices=try_all_gpus()):
    print("train on: ", devices)
    net = nn.DataParallel(net, device_ids=devices)
    net.to(devices[0])

    loss_fn = torch.nn.CrossEntropyLoss()
    optim = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=1e-4)


    metric = Accumulator(3)

    for epoch in range(num_epochs):
        net.train()
        metric.reset()
        for X, label in train_iter:
            X, label = X.to(devices[0]), label.to(devices[0])
            
            y = net(X)
            loss = loss_fn(y, label)

            print(accuracy(y, label=))

            optim.zero_grad()
            loss.sum().backward()
            optim.step()
            metric.add(loss, accuracy(y, label), 1)
        
        if (epoch+1) % 10 == 0:
            try:
                torch.save(net.state_dict(), f"./logs/epoch{epoch+1}_testacc{evaluate_test_with_GPUS(net, test_iter):3.2}_loss{metric[0]/metric[-1]:3.2}_acc{metric[1]/metric[-1]:.2}.pth")
            except:
                os.mkdir("./logs")
                torch.save(net.state_dict(), f"./logs/epoch{epoch+1}_testacc{evaluate_test_with_GPUS(net, test_iter):3.2}_loss{metric[0]/metric[-1]:3.2}_acc{metric[1]/metric[-1]:.2}.pth")


        
        


        print(epoch, loss)


def test_print_csv(net, devices=try_all_gpus()):
    preds = list()
    net.to(devices[0])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], # normalize. 归一化. 
                    [0.2023, 0.1994, 0.2010])
    ])

    test_datasets = CIFAR10_dataset(False)
    test_iter = DataLoader(test_datasets, 64)

    for X, _ in test_iter:
        y_hat = net(X.to(devices[0]))
        preds.extend(y_hat.argmax(dim=1).type(torch.int32).cpu().numpy())

    sorted_ids = list(range(1, len(test_datasets) + 1))
    sorted_ids.sort(key=lambda x: str(x))
    df = pd.DataFrame({'id': sorted_ids, 'label': preds})
    df['label'] = df['label'].apply(lambda x: test_datasets.classes[x])
    df.to_csv('submission.csv', index=False)





def parse_csv2label(train=True):
    if train:
        with open("./trainLabels.csv", "r") as f:
            return {ele[0]:ele[1] for ele in [line.strip().split(',') for line in f.readlines()][1: ]}
    else:
        with open("./sampleSubmission.csv", "r") as f:
            return {ele[0]:ele[1] for ele in [line.strip().split(',') for line in f.readlines()][1: ]}


def load_data_CIFAR10(batch_size, num_workers=28):
    # 我自己编写的CIFAR10 datasets也可以正常使用
    train_datasets, test_datasets = CIFAR10_dataset(), CIFAR10_dataset(train=False)

    # transform_train = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Resize(42), # 放大一点, CIFAR10的数据集是32, 放大到40, 可以给我们一点操作的空间, 让我能够有一些额外的操作. 
    #     transforms.RandomResizedCrop(32, scale=(0.64, 1.0), ratio=(1.0, 1.0)), 
    #     transforms.RandomHorizontalFlip(),
    #     transforms.Normalize([0.4914, 0.4822, 0.4465], # normalize. 归一化. 
    #                     [0.2023, 0.1994, 0.2010])
    # ])
    # transform_test = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.4914, 0.4822, 0.4465], # normalize. 归一化. 
    #                 [0.2023, 0.1994, 0.2010])
    # ])
    # train_datasets = torchvision.datasets.CIFAR10("./data/", True, transform=transform_train, download=True)
    # test_datasets = torchvision.datasets.CIFAR10("./data/", False, transform=transform_test, download=True)

    
    train_iter, test_iter = DataLoader(train_datasets, batch_size, True, num_workers=num_workers, drop_last=True), \
        DataLoader(test_datasets, batch_size, True, num_workers=num_workers, drop_last=False)

    return train_iter, test_iter










class Accumulator():
    def __init__(self, n) -> None:
        self.data = [0.0]*n # 初始化

    def add(self, *args):
        self.data = [a+float(b) for a, b in zip(self.data, args)]
    
    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx] 
    
    

        





class CIFAR10_dataset(Dataset):
    def __init__(self, train=True, vaild_rate=0.1) -> None:
        super().__init__()
        self.train = train
        self.transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(40), # 放大一点, CIFAR10的数据集是32, 放大到40, 可以给我们一点操作的空间, 让我能够有一些额外的操作. 
            transforms.RandomResizedCrop(32, scale=(0.64, 1.0), ratio=(1.0, 1.0)), 
            transforms.RandomHorizontalFlip(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], # normalize. 归一化. 
                            [0.2023, 0.1994, 0.2010])
        ])
        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], # normalize. 归一化. 
                        [0.2023, 0.1994, 0.2010])
        ])
        self.vaild_rate=vaild_rate
        if train:
            self.path = "./train"
            self.label_dict = parse_csv2label()

        else:
            self.path = "./test"
            self.label_dict = parse_csv2label(False)
        self.file_name_list = os.listdir(self.path)
        # self.file_path = [os.path.join(self.path, file_name) for file_name in self.file_name_list]
        
        # self.label = list(set(self.label_dict.values())) # ['bird', 'cat', 'frog', 'automobile', 'horse', 'truck', 'ship', 'deer', 'dog', 'airplane']
        self.classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']



        
    def __getitem__(self, index):
        file_name = self.file_name_list[index]
        # print(file_name)
        img = cv.imread(os.path.join(self.path, file_name))
        if self.train:
            img = self.transform_train(img)
        else:
            img = self.transform_test(img)
        classes = self.label_dict[file_name.split('.')[0].__str__()]
        # self.transform_test(np.array([self.label.index(label)]))
        return img, self.classes.index(classes)

        

    def __len__(self):
        return int(len(self.file_name_list)*self.vaild_rate)
    
    def is_current(self):
        # print(self.file_name_list)
        print(self[0])
        # print(self.label)


#@ test
if __name__ == "__main__":
    CIFAR10_dataset(train=False).is_current()
    test_datasets = torchvision.datasets.CIFAR10("./data/", False, download=True)
    print(test_datasets.classes)
    # test_iter =     DataLoader(test_datasets, 1, True, num_workers=8, drop_last=False)
    print(test_datasets[0])
    # pass

    # print(parse_csv2label())
