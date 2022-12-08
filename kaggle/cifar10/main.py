import torch
from utils import  train_cifar10, test_to_submission
from net import resnet18

BATCH_SIZE = 256
EPOCH = 200
LR = 1e-2

def main() -> None:
    net = resnet18() # 实例化. 
    # 如果使用多GPU则需要使用Dataparalleld的方式
    net = torch.nn.DataParallel(net, range(torch.cuda.device_count())).to(device=torch.device("cuda:0"))
    net.load_state_dict(torch.load('logs/epoch10_testacc0.72_loss0.65_acc0.77.pth'))

    # train_cifar10(net, LR, BATCH_SIZE, EPOCH)



    test_to_submission(net)





if __name__ == '__main__':
    main()

