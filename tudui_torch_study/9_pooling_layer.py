"""
通过torch中的API: 我们来查看一些常见的池化层函数. 

maxpool: 最大池化, 也称之为下采样.
maxUnpool: 最大池化相反的操作, 也称之为上采样.

最大池化是在kernel窗口中, 选取最大的一个元素, 然后根据stride依次遍历.

torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)


kernel_size : the size of the window to take a max over kernel大小, 也就是窗口大小. 
stride : the stride of the window. Default value is kernel_size # 步长. 默认值是kernel的大小
padding : implicit zero padding to be added on both sides # 填充,
dilation : a parameter that controls the stride of elements in the window # 卷积核中分隔大小. 
return_indices : if True, will return the max indices along with the outputs. Useful for torch.nn.MaxUnpool2d later
ceil_mode : when True, will use ceil instead of floor to compute the output shape # True, 使用ceil模型, 否则使用floor模式. 
        # ceil模式表示: ceil模式: 向上取整, 并且保留框中的情况. floor向下取整, 不进行保留. 一会我们可以看到一些效果. 

经过池化层最终的宽高仍然是:
    H = [(H+2*P-K)/S + 1]
    W = [(W+2*P-K)/S + 1]


那么, 最大池化层的目的是什么: 就是减小数据的特征, 从而减小空间开销. 保留一些特定的特征. 

""" 

# 代码测试
from nbformat import write
import torch
import torchvision
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

writer = SummaryWriter("./logs")
class MyPooling(nn.Module):
    def __init__(self) -> None:
        super().__init__() # 继承. 
        self.maxpool1 = nn.MaxPool2d(
            kernel_size=3,
            # stride=kernel_size
            # paddin=1
            ceil_mode=False, # 使用, ceil模式, 也就是保留边界. 
             
        )

    def forward(self, input):
        return self.maxpool1(input)




# 对我们的真实图片进行一个最大池化操作, 以此来看一下效果. 
def image_pooling_layer() -> None:
    CIFAR10_datasets = datasets.CIFAR10("../data/", False, transforms.ToTensor(), download=True)
    # 创建dataLoader: 
    CIFAR10_dataloader = DataLoader(CIFAR10_datasets, 64, shuffle=True)
    my_pooling = MyPooling()

    for idx, data in enumerate(CIFAR10_dataloader):
        imgs, labels = data
        print(imgs.shape) # torch.Size([64, 3, 32, 32])
        writer.add_images("pooling_before", imgs, idx)

        # 经过最大池化层处理:
        output = my_pooling(imgs) # 卷积层不会改变你的channel数, 所以我们添加到write中, 无需reshape
        print(output.shape) # torch.Size([64, 3, 10, 10])
        writer.add_images("pooling_after", output, idx)
        # 通过tensorboard我们就可以看到池化前和池化后的区别了. 
        # 可见. 我们的数据量被极大的缩减了, 这样更有利于我们的网络进行训练, 
        # 但是对于细节信息保存的不够完好, 所以在检测小物体时候可能效果不会很好(我猜的. )






def test_pooling() -> None: 
    input = torch.tensor([
        [1, 2, 0, 3, 1],
        [0, 1, 2, 3, 1],
        [1, 2, 1, 0, 0],
        [5, 2, 3, 1, 1],
        [2, 1, 0, 1, 1]
    ], dtype=torch.float32) # 最大池化, 我们一定要注意, 无法对long进行操作. 
    # 我们经常运算的时候, 都是使用float32进行运算. 

    input = input.reshape((-1, 1, 5, 5))

    out = MyPooling()(input)

    print(out)
    # ceil为True时:
    # tensor([[[[2., 3.],
    #           [5., 1.]]]])
    # 当ceil_mode为False时:
    # tensor([[[[2.]]]])


    



if __name__ == "__main__": # 检测只有本地启动时, 才进入. 
    # test_pooling()
    image_pooling_layer()








writer.close()

