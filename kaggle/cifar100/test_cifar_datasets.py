from torch.utils.data import DataLoader
import torchvision
import sys
import os
from torchvision import transforms
os.chdir(sys.path[0])

from utils import *
from EfficientNet import *
img_size = 224

train_transforme = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(256, antialias=True),
    transforms.RandomResizedCrop(img_size, scale=(0.60, 1.0), ratio=(1.0, 1.0), antialias=True),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
    transforms.Normalize([0.4914, 0.4822, 0.4465],  # normalize. 归一化.
                         [0.2023, 0.1994, 0.2010])
])

val_transforme = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(img_size, antialias=True),
    transforms.Normalize([0.4914, 0.4822, 0.4465],  # normalize. 归一化.
                         [0.2023, 0.1994, 0.2010])

])

cifar10_train_datasets = torchvision.datasets.CIFAR100(
    "~/data_fine", train=True, download=False, transform=train_transforme)
cifar10_val_datasets = torchvision.datasets.CIFAR100(
    "~/data_fine", train=False, download=False, transform=val_transforme)


lr = 0.004
batch_size = 128
num_epoch = 100


def train_val_with_cos() -> None:
    # gen_cifar10_train_val_text("cifar_data")
    train_iter = DataLoader(cifar10_train_datasets, batch_size, True, num_workers=8)
    val_iter = DataLoader(cifar10_val_datasets, batch_size, False, num_workers=8)
    

    net = efficientnet_b0(100)
    train_cos_ema(
        net, nn.CrossEntropyLoss(),
        train_iter, val_iter,
        lr, num_epoch,
        save_mode="best", test_epoch=1,
        # load_path="/home/wakinghours/programming/LiMu-DeepLearning/kaggle/cifar10_version2/logs/epoch38_testacc0.874_loss0.006_acc0.87.pth"
    )


def test_submission() -> None:
    net = efficientnet_b0(10)
    
    test_to_submission(
        net, 
        "/home/wakinghours/programming/LiMu-DeepLearning/kaggle/cifar10_version2/logs/epoch98_testacc0.908_loss0.0035_acc0.92.pth"
    )

if __name__ == "__main__":
    train_val_with_cos()
    
    # test_submission()
