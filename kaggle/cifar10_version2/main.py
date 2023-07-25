import torch, torchvision
from torch import nn


from utils import get_train_vaild_datasets, load_data_CIFAR10, train_cos_ema
from VanillaNet import VanillaNet

import os, sys

os.chdir(sys.path[0])


batch_size = 128
lr = 0.02
num_epoch = 300

def test_ema_with_cos() -> None:
    train_iter, val_iter = load_data_CIFAR10(batch_size)
    
    net = VanillaNet()
    train_cos_ema(
        net, nn.CrossEntropyLoss(),
        train_iter, val_iter,
        lr, num_epoch, 
        save_mode="best", test_epoch=1
        # load_path="logs/epoch230_testacc0.739_loss0.0028_acc0.75.pth"
    )


if __name__ == "__main__":
    test_ema_with_cos()
