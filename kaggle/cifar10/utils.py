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
from typing import List


def try_all_gpus() -> List[torch.device]:
    devices = []
    devices.extend([torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())])

    return devices if devices else "CPU"

# print(try_all_gpus())


def train(net: nn.Module, loss, optim: optim, lr, weight_decay, num_eopchs, batch_size, train_iter, test_iter, devices = try_all_gpus()):
    pass




def train_cifar10(net):
    





def parse_csv2label(train=True):
    if train:
        with open("./trainLabels.csv", "r") as f:
            return {ele[0]:ele[1] for ele in [line.strip().split(',') for line in f.readlines()][1: ]}
    else:
        with open("./sampleSubmission.csv", "r") as f:
            return {ele[0]:ele[1] for ele in [line.strip().split(',') for line in f.readlines()][1: ]}


def load_data_CIFAR10(batch_size, num_workers=4):
    train_datasets, test_datasets = CIFAR10_dataset(), CIFAR10_dataset(train=False)
    
    train_iter, test_iter = DataLoader(train_datasets, batch_size, True, num_workers=num_workers, drop_last=True), \
        DataLoader(test_datasets, batch_size, True, num_workers=num_workers, drop_last=False)

    return train_iter, test_iter

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
            self.path = "./train/train"
            self.label_dict = parse_csv2label()

        else:
            self.path = "./test/test"
            self.label_dict = parse_csv2label(False)
        self.file_name_list = os.listdir(self.path)
        # self.file_path = [os.path.join(self.path, file_name) for file_name in self.file_name_list]
        
        # self.label = list(set(self.label_dict.values())) # ['bird', 'cat', 'frog', 'automobile', 'horse', 'truck', 'ship', 'deer', 'dog', 'airplane']
        self.label = ['bird', 'cat', 'frog', 'automobile', 'horse', 'truck', 'ship', 'deer', 'dog', 'airplane']




        
    def __getitem__(self, index):
        file_name = self.file_name_list[index]
        # print(file_name)
        img = cv.imread(os.path.join(self.path, file_name))
        if self.train:
            img = self.transform_train(img)
        else:
            img = self.transform_test(img)
        label = self.label_dict[file_name.split('.')[0].__str__()]
        # self.transform_test(np.array([self.label.index(label)]))
        return img, torch.tensor(self.label.index(label))

        

    def __len__(self):
        return int(len(self.file_name_list)*self.vaild_rate)
    
    def is_current(self):
        # print(self.file_name_list)
        print(self[0])
        # print(self.label)


#@ test
if __name__ == "__main__":
    CIFAR10_dataset(train=False).is_current()
    # print(parse_csv2label())
