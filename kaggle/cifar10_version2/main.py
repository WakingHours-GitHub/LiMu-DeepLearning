import torch, torchvision
from torch import nn
from torch.utils.data import DataLoader

from utils import get_train_vaild_datasets, load_data_CIFAR10, train_cos_ema, CIFAR10_datasets, test_to_submission
from VanillaNet import VanillaNet

from EfficientNet import efficientnet_b0

import os, sys

os.chdir(sys.path[0])


batch_size = 512
lr = 0.05
num_epoch = 300

def test_ema_with_cos() -> None:
    # train_iter, val_iter = load_data_CIFAR10(batch_size)
    
    train_iter = DataLoader(
        CIFAR10_datasets(), batch_size, True, num_workers=4
    )
    
    # net = VanillaNet()
    net = efficientnet_b0(10)
    train_cos_ema(
        net, nn.CrossEntropyLoss(),
        train_iter, None,
        lr, num_epoch, 
        save_mode="epoch", test_epoch=1, 
        # load_path="logs/epoch89_testacc0.615_loss0.0095_acc0.58.pth"
    )


def test_submission() -> None:
    net = efficientnet_b0(10)
    
    test_to_submission(
        net, 
        "/home/wakinghours/programming/LiMu-DeepLearning/kaggle/cifar10_version2/logs/epoch221_testacc 0.0_loss0.0031_acc0.73.pth"
    )


if __name__ == "__main__":
    test_ema_with_cos()
    # test_submission()
