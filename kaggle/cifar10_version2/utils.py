import torch
import torchvision
from torchvision import transforms
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split  # 导入工具。
import cv2 as cv
import os
import sys
from typing import List
import math

os.chdir(sys.path[0])
join = os.path.join


def try_all_gpus() -> List[torch.device]:
    devices = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
    return devices if devices else torch.device("cpu")






class Accumulator():
    def __init__(self, n) -> None:
        self.data = [0.0] * n  # 初始化

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
    
    
    
    
def accuracy(y_hat: torch.Tensor, y: torch.Tensor) -> float:
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:  # 二维, 并且列数要大于1.
        y_hat = y_hat.argmax(dim=1)  # 返回指定轴上值最大的索引. 这里是按照行为单位, 所以dim=1
    cmp = y_hat.type(y.dtype) == y # 转换为同一种类型.
    return float(cmp.type(y.dtype).sum()) # 这里求和. 所以需要在外面减去. 
    # 因为我们不知道这一批量具体是多少, 所以我们在Accumulate中进行计算精度的操作. 

def train_cos(
    net: nn.Module, loss_fn: nn.Module,
    train_iter: DataLoader, test_iter: DataLoader,
    lr, num_epoch,
    momentum=0.937, weight_decay=5e-4,
    load_path: str = None, devices=try_all_gpus(),
):
    eps = 0.35
    net = nn.DataParallel(net, devices).to(devices[0])
    loss_fn.to(devices[0])
    if load_path:
        print("load net parameters: ", load_path)
        net.load_state_dict(torch.load(load_path))  # load parameters for net module.
    trainer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    def lf(x): return ((1 + math.cos(x * math.pi / num_epoch)) / 2) * (1 - eps) + eps
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer=trainer,  # 优化器.
        lr_lambda=lf,  # 学习率函数.
    )  # 根据lambda自定义学习率.

    metric = Accumulator(3)
    trainer.zero_grad()  # empty.
    # train:
    print("train on:", devices)
    for epoch in range(num_epoch):
        net.train()
        metric.reset()  # 重置
        # train a epoch.
        for i, (x, labels) in enumerate(train_iter):
            x, labels = x.to(devices[0]), labels.to(devices[0])
            y_hat = net(x)
            loss = loss_fn(y_hat, labels)

            trainer.zero_grad()  # first empty gradient
            loss.sum().backward()  # than calculate graient by backwoard (pro)
            trainer.step()  # update weight.

            metric.add(loss.item(), accuracy(y_hat, labels), labels.shape[0])

        scheduler.step()  # we only update scheduler.

        # evaluate
        if (epoch + 1) % 10 == 0:
            test_accuracy = evaluate_test_with_GPUS(net, test_iter)
            print(epoch + 1, "test acc:", test_accuracy, "train loss:",
                  metric[0] / metric[-1], "train acc:", metric[1] / metric[-1])

            try:
                torch.save(
                    net.state_dict(),
                    f"./logs/epoch{epoch+1}_testacc{test_accuracy:4.3}_loss{metric[0]/metric[-1]:3.2}_acc{metric[1]/metric[-1]:.2}.pth")
            except BaseException:
                os.mkdir("./logs")
                torch.save(
                    net.state_dict(),
                    f"./logs/epoch{epoch+1}_testacc{test_accuracy:4.3}_loss{metric[0]/metric[-1]:3.2}_acc{metric[1]/metric[-1]:.2}.pth")









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
