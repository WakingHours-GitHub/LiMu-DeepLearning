import torch
import torchvision
from torchvision import transforms
from torchsummary import summary
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split  # 导入工具。
import cv2 as cv
import os
import sys
from typing import List
import math
import pandas as pd


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



def test_to_submission(net: nn.Module, load_path: str = None, devices=try_all_gpus()):
    net = nn.DataParallel(net, devices).to(devices[0])  # 如果训练时, 我们使用了DataParallel, 那么在测试时, 我们也一定要使用.
    # 然后才能load参数.
    net.load_state_dict(torch.load(load_path))

    sorted_ids, preds = [], []

    datasets = CIFAR10_datasets(is_train=False)
    # print(datasets.__len__())
    test_iter = DataLoader(
        datasets,
        5000,
        False,
        num_workers=14,
        drop_last=False,
    )
    print(len(test_iter))

    for i, (X, file_name) in enumerate(test_iter):
        # print(file_name)
        print(i)
        y_hat = net(X.to(devices[0]))
        sorted_ids.extend([int(i) for i in file_name])
        preds.extend(
            y_hat.argmax(dim=1).type(torch.int32).cpu().numpy()
        )
        # break


    # print(sorted_ids)

    sorted_cord = sorted(zip(sorted_ids, preds), key=lambda x: x[0])
    result = zip(*sorted_cord)
    sorted_ids, preds = [list(x) for x in result]

    df = pd.DataFrame({'id': sorted_ids, 'label': preds})
    df['label'] = df['label'].apply(lambda x: datasets.classes[x])
    df.to_csv('./submission.csv', index=False)



def evaluate_test_with_GPUS_ema(ema, net: nn.Module, test_iter) -> float:
    if isinstance(net, nn.Module):
        net.eval()  # set module on evaluation mode
        ema.apply_shadow()

    device = next(iter(net.parameters())).device  # 拿出来一个参数的device
    matrix = Accumulator(2)
    with torch.no_grad():
        for X, labels in test_iter:
            X, labels = X.to(device), labels.to(device)
            matrix.add(accuracy(net(X), labels), labels.shape[0])
            # print(net(X).shape)
            # matrix.add(accuracy(net(X), labels), 1)  # here, accuracy use 'sum()', so add BS.
            # here, accuracy used 'mean()' so, add 1.
    ema.restore()
    return matrix[0] / matrix[1]  # 均值.


class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


def accuracy(y_hat: torch.Tensor, y: torch.Tensor) -> float:
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:  # 二维, 并且列数要大于1.
        y_hat = y_hat.argmax(dim=1)  # 返回指定轴上值最大的索引. 这里是按照行为单位, 所以dim=1
    cmp = y_hat.type(y.dtype) == y  # 转换为同一种类型.
    return float(cmp.type(y.dtype).sum())  # 这里求和. 所以需要在外面减去.
    # 因为我们不知道这一批量具体是多少, 所以我们在Accumulate中进行计算精度的操作.


def train_cos_ema(
    net: nn.Module, loss_fn: nn.Module,
    train_iter: DataLoader, test_iter: DataLoader,
    lr, num_epoch,
    momentum=0.937, weight_decay=5e-4,
    load_path: str = None, save_mode: str = "epoch", test_epoch:int=10, 
    devices=try_all_gpus(),
):
    assert save_mode in ("epoch", "best"), "[ERROR]: save_mode must be is epoch or best"
    
    eps = 0.35
    net = nn.DataParallel(net, devices).to(devices[0])

    ema = EMA(net, 0.999)
    ema.register()  # 注册.
    
    best_test_accuracy = 0

    loss_fn.to(devices[0])
    if load_path:
        print("load net parameters: ", load_path)
        net.load_state_dict(torch.load(load_path))  # load parameters for net module.
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    
    # from EfficientNet.
    # optimizer = torch.optim.RMSprop(net.parameters(), lr=lr, alpha=0.99, eps=1e-8, weight_decay=0.9, momentum=0.9)
    def lf(x): return ((1 + math.cos(x * math.pi / num_epoch)) / 2) * (1 - eps) + eps
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer=optimizer,  # 优化器.
        lr_lambda=lf,  # 学习率函数.
    )  # 根据lambda自定义学习率.

    metric = Accumulator(3)
    optimizer.zero_grad()  # empty.
    
    # train:
    # print(next(iter(train_iter))[0].shape[1: ])
    summary(net, input_size=next(iter(train_iter))[0].shape[1: ])
    print("train on:", devices)
    print("save mode: ", save_mode)
    for epoch in range(num_epoch):
        net.train()
        metric.reset()  # 重置
        # train a epoch.
        for i, (x, labels) in enumerate(train_iter):
            x, labels = x.to(devices[0]), labels.to(devices[0])
            y_hat = net(x)
            loss = loss_fn(y_hat, labels)

            optimizer.zero_grad()  # first empty gradient
            loss.sum().backward()  # than calculate graient by backwoard (pro)
            optimizer.step()  # update weight.
            ema.update()

            metric.add(loss.item(), accuracy(y_hat, labels), labels.shape[0])

        scheduler.step()  # we only update scheduler.

        # evaluate
        if (epoch + 1) % test_epoch == 0:
            if test_iter != None:
                test_accuracy = evaluate_test_with_GPUS_ema(ema, net, test_iter)
            else: # test_iter = None. 
                test_accuracy = 0.0
            print(epoch + 1, "test acc:", test_accuracy, "train loss:",
                  metric[0] / metric[-1], "train acc:", metric[1] / metric[-1])
            if save_mode == "epoch":
                try:
                    torch.save(
                        net.state_dict(),
                        f"./logs/epoch{epoch+1}_testacc{test_accuracy:4.3}_loss{metric[0]/metric[-1]:3.2}_acc{metric[1]/metric[-1]:.2}.pth")
                except BaseException as e:
                    print(e)
                    os.mkdir("./logs")
                    torch.save(
                        net.state_dict(),
                        f"./logs/epoch{epoch+1}_testacc{test_accuracy:4.3}_loss{metric[0]/metric[-1]:3.2}_acc{metric[1]/metric[-1]:.2}.pth")
            elif save_mode == "best":
                if best_test_accuracy < test_accuracy:
                    best_test_accuracy = test_accuracy  # update current.
                    print("best: ", test_accuracy)
                    try:
                        torch.save(net.state_dict(), f"./logs/epoch{epoch+1}_testacc{test_accuracy:4.3}_loss{metric[0]/metric[-1]:3.2}_acc{metric[1]/metric[-1]:.2}.pth")
                    except BaseException as e:
                        os.mkdir("./logs")
                        torch.save(net.state_dict(), f"./logs/epoch{epoch+1}_testacc{test_accuracy:4.3}_loss{metric[0]/metric[-1]:3.2}_acc{metric[1]/metric[-1]:.2}.pth")
                        

def load_data_CIFAR10(batch_size, num_workers=14):
    # 我自己编写的CIFAR10 datasets也可以正常使用
    train_datasets, test_datasets = get_train_vaild_datasets(vaild_rate=0)

    train_iter, test_iter = DataLoader(train_datasets, batch_size, True, num_workers=num_workers, drop_last=True), \
        DataLoader(test_datasets, batch_size, True, num_workers=num_workers, drop_last=False)

    return train_iter, test_iter


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
                transforms.Resize(40, antialias=True),
                transforms.RandomResizedCrop(32, scale=(0.60, 1.0), ratio=(1.0, 1.0), antialias=True),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
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
        # print(file_name)
        img = self.transforme(cv.imread(join(self.path, file_name)))
        if self.is_train:
            label = self.labels_list[int(file_name.split(".")[0])]
            label = self.classes.index(label)
        else:
            label = file_name.split(".")[0]

        return img, label

    def __len__(self):
        return len(self.file_name_list)

    def parse_csv2label(self):
        with open(join(self.root_path, "trainLabels.csv"), "r") as f:
            # return {ele[0]: ele[1] for ele in [line.strip().split(',') for line in f.readlines()]}
            return [ele[1] for ele in [line.strip().split(',') for line in f.readlines()]]


if __name__ == "__main__":
    train, val = get_train_vaild_datasets()
    print(train[0])
    print(train.__len__())
    print(val.__len__())
    
    
    train, val = load_data_CIFAR10(20)
    for x, labels in train:
        print(x.shape)
        print(labels)
        break
