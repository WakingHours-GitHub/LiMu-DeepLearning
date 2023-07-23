import torch
import torchvision
from torchvision import transforms
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split  # 导入工具。
import cv2 as cv
import os
import sys
from typing import List


os.chdir(sys.path[0])
join = os.path.join


def try_all_gpus() -> List[torch.device]:
    devices = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
    return devices if devices else torch.device("cpu")










def get_train_vaild_datasets(vaild_rate=0.1):
    train_datasets = CIFAR10_datasets()
    len = train_datasets.__len__()

    return random_split(train_datasets, [int(len - len * vaild_rate), int(len * vaild_rate)])


















class CIFAR10_datasets(Dataset):
    def __init__(self, is_train=True) -> None:
        super().__init__()
        self.is_train = is_train
        # self.root_path = "/home/wakinghours/data_fine/cifar-10" # absolute path
        self.root_path = "./cifar_data"
        if self.is_train:
            self.path = join(self.root_path, "train")
            self.transforme = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(42, antialias=True),
                transforms.RandomResizedCrop(32, scale=(0.60, 1.0), ratio=(1.0, 1.0), antialias=True),
                transforms.RandomHorizontalFlip(),
                # transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
                transforms.Normalize([0.4914, 0.4822, 0.4465],  # normalize. 归一化.
                                     [0.2023, 0.1994, 0.2010])
            ])
        else:
            self.path = join(self.root_path, "test")
            self.transforme = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465],  # normalize. 归一化.
                                     [0.2023, 0.1994, 0.2010])
            ])

        self.labels_list = self.parse_csv2label()
        self.classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

        self.file_name_list = os.listdir(self.path)
        
    def __getitem__(self, index):
        file_name = self.file_name_list[index]
        print(file_name)
        img = self.transforme(cv.imread(join(self.path, file_name)))
        if self.is_train:
            label = self.labels_list[int(file_name.split(".")[0])]
        else:
            label = None
        
        return img, label

    def __len__(self):
        return len(self.file_name_list)

    def parse_csv2label(self):
        with open(join(self.root_path, "trainLabels.csv"), "r") as f:
            # return {ele[0]: ele[1] for ele in [line.strip().split(',') for line in f.readlines()]}
            return [ele[1] for ele in [line.strip().split(',') for line in f.readlines()]]


if __name__ == "__main__":
    train, val = get_train_vaild_datasets() 
    print(train.__len__())
    print(val.__len__())
