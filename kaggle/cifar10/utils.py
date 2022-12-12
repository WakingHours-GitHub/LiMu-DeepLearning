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


def try_all_GPUS() -> List[torch.device]:
    devices = []
    devices.extend([torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())])
    return devices if devices else "cpu"







def test_to_submission(net:nn.Module, load_path:str=None,devices=try_all_GPUS()):
    net = nn.DataParallel(net, devices).to(devices[0]) # 如果训练时, 我们使用了DataParallel, 那么在测试时, 我们也一定要使用. 
    # 然后才能load参数. 
    net.load_state_dict(torch.load(load_path))

    sorted_ids, preds = [], []

    datasets = CIFAR10_datasets(is_train=False)
    # print(datasets.__len__())
    test_iter = DataLoader(
        datasets,
        5000, 
        False,
        num_workers=28, 
        drop_last=False,

    )
    print(len(test_iter))
    
    for i, (X, file_name) in enumerate(test_iter):
        # print(file_name)
        print(i)
        y_hat=net(X.to(devices[0]))
        sorted_ids.extend([int(i) for i in file_name])
        preds.extend(
            y_hat.argmax(dim=1).type(torch.int32).cpu().numpy()
        )

        # break


    # print(sorted_ids)

    sorted_cord = sorted(zip(sorted_ids, preds), key=lambda x:x[0])
    result = zip(*sorted_cord)
    sorted_ids, preds = [list(x) for x in result]

    df = pd.DataFrame({'id': sorted_ids, 'label': preds})
    df['label'] = df['label'].apply(lambda x: datasets.classes[x])
    df.to_csv('./submission.csv', index=False)


def train(net: nn.Module, loss_fn, train_iter, vaild_iter, lr, num_epochs, 
        lr_period, lr_decay, weight_decay ,devices=try_all_GPUS(), load_path:str = None):
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
            loss.mean().backward()
            trainer.step()

            metric.add(loss.item(), accuracy(y, labels), 1) # 因为都是使用的平均值, 所以这里加1
            # 如果loss, 和accuracy使用的是sum. 那么这里就要改成labels.shape[0]也就是多少个批量.
        scheduler.step() # 每轮结束后我们要更新一下这个scheduler. 



        # save net paramters: 
        if (epoch+1) % 10 == 0:
            test_accuracy = evaluate_test_with_GPUS(net, vaild_iter)
            print(epoch, "test acc:", test_accuracy, "train loss:", metric[0]/metric[-1], "train acc:", metric[1]/metric[-1])

            try:
                torch.save(net.state_dict(), f"./logs/epoch{epoch+1}_testacc{test_accuracy:3.2}_loss{metric[0]/metric[-1]:3.2}_acc{metric[1]/metric[-1]:.2}.pth")
            except:
                os.mkdir("./logs")
                torch.save(net.state_dict(), f"./logs/epoch{epoch+1}_testacc{test_accuracy:3.2}_loss{metric[0]/metric[-1]:3.2}_acc{metric[1]/metric[-1]:.2}.pth")


            



    



def train_cifar10(net, lr, batch_size, num_epoch):
    print("train!!!")
    devices = try_all_GPUS()
    train_iter, vaild_iter = load_data_CIFAR10(batch_size)

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





def evaluate_test_with_GPUS(net: nn.Module, test_iter):
    if isinstance(net, nn.Module):
        net.eval() # set module on evaluation mode

    device = next(iter(net.parameters())).device # 拿出来一个参数的device
    matrix = Accumulator(2)
    with torch.no_grad():
        for X,  label in test_iter:
            X, label = X.to(device), label.to(device)
            # matrix.add(accuracy(net(X), label), label.shape[0])
            matrix.add(accuracy(net(X), label), 1) # here, accuracy used 'mean()' so, add 1.
    return matrix[0] / matrix[1] # 均值.
    



def accuracy(y_hat:torch.Tensor, label:torch.Tensor):
    return (y_hat.argmax(dim=1) == label).float().mean()


class Accumulator():
    def __init__(self, n) -> None:
        self.data = [0.0]*n # 初始化

    def add(self, *args):
        self.data = [a+float(b) for a, b in zip(self.data, args)]
    
    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx] 



def load_data_CIFAR10(batch_size, num_workers=28):
    # 我自己编写的CIFAR10 datasets也可以正常使用
    train_datasets, test_datasets = get_train_vaild_datasets()

    train_iter, test_iter = DataLoader(train_datasets, batch_size, True, num_workers=num_workers, drop_last=True), \
        DataLoader(test_datasets, batch_size, True, num_workers=num_workers, drop_last=False)

    return train_iter, test_iter

def get_train_vaild_datasets(vaild_rate=0.1):
    train_datasets = CIFAR10_datasets()
    len = train_datasets.__len__()

    return random_split(train_datasets, [int(len-len*vaild_rate), int(len*vaild_rate)])

    

class CIFAR10_datasets(Dataset):
    def __init__(self, is_train=True) -> None:
        super().__init__()
        self.is_train=is_train
        if self.is_train:
            self.path = "./train"
        else:
            self.path = "./test"
            
        self.label_dict = self.parse_csv2label() # return
        self.classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

        self.file_name_list = os.listdir(self.path)

        self.transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(42), # 放大一点, CIFAR10的数据集是32, 放大到40, 可以给我们一点操作的空间, 让我能够有一些额外的操作. 
            transforms.RandomResizedCrop(32, scale=(0.60, 1.0), ratio=(1.0, 1.0)), 
            transforms.RandomHorizontalFlip(),
            # transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
            transforms.Normalize([0.4914, 0.4822, 0.4465], # normalize. 归一化. 
                            [0.2023, 0.1994, 0.2010])
        ])
        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], # normalize. 归一化. 
                        [0.2023, 0.1994, 0.2010])
        ])



    def __getitem__(self, index):
        if self.is_train:
            file_name = self.file_name_list[index]
            label = self.classes.index(self.label_dict[file_name.split('.')[0]])
            img = self.transform_train(cv.imread(os.path.join(self.path, file_name)))
            
            return img, label
        else:
            file_name = self.file_name_list[index]
            try:
                img = self.transform_test(cv.imread(os.path.join(self.path, file_name)))
            except Exception as e:
                print(e)
                print(file_name)
                return torch.zeros(size=(3, 32, 32)), 0
            return img, file_name.split(".")[0]

    def __len__(self):
        return len(self.file_name_list)


    def is_current(self):
        print(self[0])
    

    def parse_csv2label(self):
            with open("./trainLabels.csv", "r") as f:
                return {ele[0]:ele[1] for ele in [line.strip().split(',') for line in f.readlines()][1: ]}




class CIFAR10_Train_Test(Dataset):
    def __init__(self, is_train=True) -> None:
        super().__init__()
        self.is_train=is_train
        if self.is_train:
            self.path = "./train"
        else:
            self.path = "./test"
            
        self.label_dict = self.parse_csv2label() # return
        self.classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

        self.file_name_list = os.listdir(self.path)

        self.transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(40), # 放大一点, CIFAR10的数据集是32, 放大到40, 可以给我们一点操作的空间, 让我能够有一些额外的操作. 
            transforms.RandomResizedCrop(32, scale=(0.64, 1.0), ratio=(1.0, 1.0)), 
            transforms.RandomHorizontalFlip(),
            # transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
            transforms.Normalize([0.4914, 0.4822, 0.4465], # normalize. 归一化. 
                            [0.2023, 0.1994, 0.2010])
        ])
        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], # normalize. 归一化. 
                        [0.2023, 0.1994, 0.2010])
        ])



    def __getitem__(self, index):
        file_name = self.file_name_list[index]
        label = self.classes.index(self.label_dict[file_name.split('.')[0]])

        if self.is_train:
            img = self.transform_train(cv.imread(os.path.join(self.path, file_name)))
        else:
            img = self.transform_test(cv.imread(os.path.join(self.path, file_name)))
        return img, label


    def __len__(self):
        return len(self.file_name_list)


    def is_current(self):
        print(self[0])

    def load_CIFAR10_train_test_dataloader(self, batch_size, num_workers=28):
        return DataLoader(
            CIFAR10_Train_Test(True),
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True
        ), DataLoader(
            CIFAR10_Train_Test( ),
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers, 
        )

    

    def parse_csv2label(self):
        if self.is_train:
            with open("./trainLabels.csv", "r") as f:
                return {ele[0]:ele[1] for ele in [line.strip().split(',') for line in f.readlines()][1: ]}
        else:
            with open("./test_label.csv", "r") as f:
                return {ele[0]:ele[1] for ele in [line.strip().split(',') for line in f.readlines()][1: ]}
        

if __name__ == '__main__':
    print(get_train_vaild_datasets())
