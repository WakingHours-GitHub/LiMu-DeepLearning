import torch, torchvision
from torch import nn
from torch.utils.data import DataLoader

from utils import get_train_vaild_datasets, load_data_CIFAR10, train_cos_ema, CIFAR10_datasets, test_to_submission
from VanillaNet import VanillaNet

from utils import get_train_vaild_datasets

from EfficientNet import efficientnet_b0, efficientnet_b3

import os, sys

os.chdir(sys.path[0])


batch_size = 64
lr = 0.02
num_epoch = 40

def test_ema_with_cos() -> None:
    # train_iter, val_iter = load_data_CIFAR10(batch_size)
    
    train_iter = DataLoader(
        CIFAR10_datasets(), batch_size, True, num_workers=14
    )
    
    # net = VanillaNet()
    net = efficientnet_b3(10)
    train_cos_ema(
        net, nn.CrossEntropyLoss(),
        train_iter, None,
        lr, num_epoch, 
        save_mode="epoch", test_epoch=1, 
        # load_path="/home/wakinghours/programming/LiMu-DeepLearning/kaggle/cifar10_version2/logs/epoch70_testacc 0.0_loss0.011_acc0.76.pth"
    )
    
def train_val_with_cos() -> None:
    # gen_cifar10_train_val_text("cifar_data")
    train_iter, val_iter = get_train_vaild_datasets(batch_size, num_workers=5)
    
    net = efficientnet_b0(10)
    train_cos_ema(
        net, nn.CrossEntropyLoss(),
        train_iter,  val_iter,
        lr, num_epoch, 
        save_mode="best", test_epoch=1, 
        # load_path="/home/wakinghours/programming/LiMu-DeepLearning/kaggle/cifar10_version2/logs/epoch70_testacc 0.0_loss0.011_acc0.76.pth"
    )


def test_submission() -> None:
    net = efficientnet_b3(10)
    
    test_to_submission(
        net, 
        "/home/wakinghours/programming/LiMu-DeepLearning/kaggle/cifar10_version2/logs/epoch17_testacc 0.0_loss0.0093_acc0.8.pth"
    )


if __name__ == "__main__":
    train_val_with_cos()
    # test_ema_with_cos()
    # test_submission()
