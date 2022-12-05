import torch
from utils import load_data_CIFAR10, train_cifar10
from net import resnet18

BATCH_SIZE = 64
LR = 1e-1

def main() -> None:
    net = resnet18()


    train_iter, test_iter = load_data_CIFAR10(BATCH_SIZE)

    train_cifar10(net, LR, 100, train_iter, test_iter, torch.device("cuda:0"))






if __name__ == '__main__':
    main()

