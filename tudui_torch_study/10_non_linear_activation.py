"""
在神经网络中, 我们使用非线性激活来引入一些非线性因子.
最长使用的就是ReLU()因为计算简单, 并且效果不错: H=max(0, X)




"""
import os
print(os.getcwd())
import torch
from torch import nn
import torchvision
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("./logs")


class ReluTest(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.relu = nn.ReLU() # default: inplace = False

        self.sigmoid = nn.Sigmoid()
    
    def forward(self, X):
        # return self.relu(X)
        return self.sigmoid(X)
    

def image_activation() -> None:
    CIFAR10_test = datasets.CIFAR10(
        root="./data",
        train=False,
        transform=transforms.ToTensor(),
        download=True,

    )

    CIFAR10_dataloader = DataLoader(
        CIFAR10_test,
        batch_size=64,
        shuffle=True,
        num_workers=4,
    )
    
    for idx, (images, labels) in enumerate(CIFAR10_dataloader):
        print(images.shape)
        writer.add_images("simoid activation", images, idx)

        output = ReluTest()(images)

        writer.add_images("simoil avtivation after", output, idx)
        

        break

    writer.close()
    




def relu_test() -> None:
    input = torch.tensor([
        [1, -0.5],
        [-1, 3]
    ])

    input = input.reshape((-1, 1, 2, 2))

    output = ReluTest()(input)

    print(output)
    # tensor([[[[1., 0.],  # 可见, 小于0的地方已经为0了.
        #   [0., 3.]]]])

if __name__ == "__main__":
    # relu_test()
    image_activation()





writer.close()





