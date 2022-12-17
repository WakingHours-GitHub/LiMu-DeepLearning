import torch
from utils import  train_cifar10, test_to_submission, train, load_data_CIFAR10, CIFAR10_Train_Test
from net import resnet18

BATCH_SIZE = 512
EPOCH = 300
LR = 2e-1

def main() -> None:
    net = resnet18() # 实例化. 
    # 如果使用多GPU则需要使用Dataparalleld的方式
    # train_iter, vaild_iter = load_data_CIFAR10(BATCH_SIZE)
    CIFAR10 = CIFAR10_Train_Test()
    train_iter, test_iter = CIFAR10.load_CIFAR10_train_test_dataloader(batch_size=BATCH_SIZE)

    # train_cifar10(net, LR, BATCH_SIZE, EPOCH)

    train(
        net,
        torch.nn.CrossEntropyLoss(),
        train_iter, test_iter, 
        LR, EPOCH, 10, 0.95, 5e-3,   
        # load_path="logs/epoch200_testacc0.85_loss0.33_acc0.89.pth" 
    )


    test_to_submission(net)


if __name__ == "__main__":
    main()