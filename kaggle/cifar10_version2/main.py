import torch, torchvision
from torch import nn


from utils import get_train_vaild_datasets, load_data_CIFAR10, train_cos_ema
from VanillaNet import VanillaNet

batch_size = 128
lr = 0.01
num_epoch = 10

def test_ema_with_cos() -> None:
    train_iter, val_iter = load_data_CIFAR10(batch_size)
    
    net = VanillaNet()
    train_cos_ema(
        net, nn.CrossEntropyLoss(),
        train_iter, val_iter,
        lr, num_epoch
        
    )


if __name__ == "__main__":
    test_ema_with_cos()
