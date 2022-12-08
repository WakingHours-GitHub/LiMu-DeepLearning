import torch
from utils import  train_cifar10, test_to_submission, train, load_data_CIFAR10
from net import resnet18

BATCH_SIZE = 256
EPOCH = 200
LR = 1e-2

def main() -> None:
    net = resnet18() # 实例化. 
    # 如果使用多GPU则需要使用Dataparalleld的方式
    train_iter, vaild_iter = load_data_CIFAR10(BATCH_SIZE)

    # train_cifar10(net, LR, BATCH_SIZE, EPOCH)
    # train(
    #     net,
    #     torch.nn.CrossEntropyLoss(),
    #     train_iter, vaild_iter, 
    #     LR, EPOCH, 10, 0.95, 5e-3,    
    # )



    test_to_submission(net, "logs/epoch50_testacc0.77_loss0.38_acc0.87.pth")





if __name__ == '__main__':
    main()

